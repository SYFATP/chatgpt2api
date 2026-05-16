from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from services.config import DATA_DIR, config
from services.content_filter import request_text
from services.log_service import LOG_TYPE_CALL, log_service
from services.protocol import openai_v1_image_edit, openai_v1_image_generations

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TASK_STATUS_DELETED = "deleted"
TASK_STATUS_DELETED_RUNNING = "deleted/running"
TASK_STATUS_DELETED_SUCCESS = "deleted/success"
TASK_STATUS_DELETED_FAILED = "deleted/failed"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:26], fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _collect_image_urls(data: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _status_with_deleted(task_status: str, results_deleted: bool) -> str:
    if not results_deleted:
        return task_status
    if task_status == TASK_STATUS_SUCCESS:
        return TASK_STATUS_DELETED_SUCCESS
    if task_status == TASK_STATUS_ERROR:
        return TASK_STATUS_DELETED_FAILED
    if task_status == TASK_STATUS_RUNNING:
        return TASK_STATUS_DELETED_RUNNING
    return TASK_STATUS_DELETED


def _task_status_label(mode: str) -> str:
    return "图生图任务" if mode == "edit" else "文生图任务"


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "status": task.get("status"),
        "mode": task.get("mode"),
        "model": task.get("model"),
        "size": task.get("size"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if task.get("data") is not None:
        item["data"] = task.get("data")
    if task.get("error"):
        item["error"] = task.get("error")
    return item


class ImageTaskService:
    def __init__(
        self,
        path: Path,
        *,
        generation_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_generations.handle,
        edit_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_edit.handle,
        retention_days_getter: Callable[[], int] | None = None,
    ):
        self.path = path
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks = self._load_locked()
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()

    def submit_generation(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        base_url: str,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": 1,
            "size": size,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="generate", payload=payload)

    def submit_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        base_url: str,
        images: list[tuple[bytes, str, str]],
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "images": images,
            "model": model,
            "n": 1,
            "size": size,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="edit", payload=payload)

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested_ids = [_clean(task_id) for task_id in task_ids if _clean(task_id)]
        with self._lock:
            if self._cleanup_locked():
                self._save_locked()
            items = []
            missing_ids = []
            for task_id in requested_ids:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                else:
                    items.append(_public_task(task))
            if not requested_ids:
                items = [
                    _public_task(task)
                    for task in self._tasks.values()
                    if task.get("owner_id") == owner
                ]
                items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
                missing_ids = []
            return {"items": items, "missing_ids": missing_ids}

    def mark_results_deleted(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        updated_ids: list[str] = []
        missing_ids: list[str] = []
        with self._lock:
            changed = False
            for task_id in [_clean(task_id) for task_id in task_ids if _clean(task_id)]:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                    continue
                task["results_deleted"] = True
                task["deleted_at"] = _now_iso()
                task["status"] = _status_with_deleted(_clean(task.get("status")), True)
                updated_ids.append(task_id)
                changed = True
                self._sync_log_locked(task)
            if changed:
                self._save_locked()
        return {"updated_ids": updated_ids, "missing_ids": missing_ids}

    def _base_log_status(self, task: dict[str, Any]) -> str:
        status = _clean(task.get("status"))
        if status == TASK_STATUS_DELETED_RUNNING:
            return TASK_STATUS_RUNNING
        if status == TASK_STATUS_DELETED_SUCCESS:
            return TASK_STATUS_SUCCESS
        if status == TASK_STATUS_DELETED_FAILED:
            return TASK_STATUS_ERROR
        if status == TASK_STATUS_DELETED:
            return TASK_STATUS_QUEUED
        return status

    def _build_log_detail(self, task: dict[str, Any]) -> dict[str, Any]:
        detail = {
            "task_id": task.get("id"),
            "key_id": task.get("key_id"),
            "key_name": task.get("key_name"),
            "role": task.get("role"),
            "endpoint": task.get("endpoint"),
            "model": task.get("model"),
            "size": task.get("size"),
            "mode": task.get("mode"),
            "started_at": task.get("started_at") or task.get("created_at"),
            "status": task.get("status"),
            "results_deleted": bool(task.get("results_deleted")),
        }
        if task.get("deleted_at"):
            detail["deleted_at"] = task.get("deleted_at")
            detail["deletion_source"] = "frontend"
        if task.get("ended_at"):
            detail["ended_at"] = task.get("ended_at")
        if task.get("duration_ms") is not None:
            detail["duration_ms"] = task.get("duration_ms")
        request_preview = _clean(task.get("request_text"))
        if request_preview:
            detail["request_text"] = request_preview
        error = _clean(task.get("error"))
        if error:
            detail["error"] = error
        urls = task.get("urls")
        if isinstance(urls, list) and urls:
            detail["urls"] = list(dict.fromkeys(str(url) for url in urls if str(url or "").strip()))
        return detail

    def _sync_log_locked(self, task: dict[str, Any]) -> None:
        log_id = _clean(task.get("log_id"))
        if not log_id:
            return
        try:
            log_service.update(log_id, summary=_task_status_label(_clean(task.get("mode"))), detail=self._build_log_detail(task))
        except Exception:
            pass

    def _create_task_log(self, task: dict[str, Any]) -> str:
        try:
            return log_service.add(
                LOG_TYPE_CALL,
                _task_status_label(_clean(task.get("mode"))),
                self._build_log_detail(task),
            )
        except Exception:
            return ""

    def _set_task_status(self, task: dict[str, Any], status: str, *, error: str = "", data: list[Any] | None = None, ended_at: str | None = None, duration_ms: int | None = None) -> None:
        task["base_status"] = status
        task["status"] = _status_with_deleted(status, bool(task.get("results_deleted")))
        task["updated_at"] = _now_iso()
        if error:
            task["error"] = error
        else:
            task.pop("error", None)
        if data is not None:
            task["data"] = data
            task["urls"] = _collect_image_urls(data)
        if ended_at:
            task["ended_at"] = ended_at
        if duration_ms is not None:
            task["duration_ms"] = duration_ms
        self._sync_log_locked(task)

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        mode: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        now = _now_iso()
        should_start = False
        with self._lock:
            cleaned = self._cleanup_locked()
            task = self._tasks.get(key)
            if task is not None:
                if cleaned:
                    self._save_locked()
                return _public_task(task)
            task = {
                "id": task_id,
                "owner_id": owner,
                "key_id": _clean(identity.get("id")),
                "key_name": _clean(identity.get("name")),
                "role": _clean(identity.get("role")),
                "status": TASK_STATUS_QUEUED,
                "base_status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": _clean(payload.get("model"), "gpt-image-2"),
                "size": _clean(payload.get("size")),
                "endpoint": "/v1/images/edits" if mode == "edit" else "/v1/images/generations",
                "request_text": request_text(payload.get("prompt")),
                "created_at": now,
                "updated_at": now,
                "started_at": now,
                "results_deleted": False,
            }
            task["log_id"] = self._create_task_log(task)
            self._tasks[key] = task
            self._save_locked()
            should_start = True

        if should_start:
            thread = threading.Thread(
                target=self._run_task,
                args=(key, mode, payload, dict(identity), _clean(payload.get("model"), "gpt-image-2")),
                name=f"image-task-{task_id[:16]}",
                daemon=True,
            )
            thread.start()
        return _public_task(task)

    def _run_task(
        self,
        key: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
    ) -> None:
        started = time.time()
        self._update_task(key, status=TASK_STATUS_RUNNING, error="", started_at=datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"))
        try:
            handler = self.edit_handler if mode == "edit" else self.generation_handler
            result = handler(payload)
            if not isinstance(result, dict):
                raise RuntimeError("image task returned streaming result unexpectedly")
            data = result.get("data")
            if not isinstance(data, list) or not data:
                message = _clean(result.get("message")) or "image task returned no image data"
                raise RuntimeError(message)
            ended_at = _now_iso()
            duration_ms = int((time.time() - started) * 1000)
            self._update_task(
                key,
                status=TASK_STATUS_SUCCESS,
                data=data,
                error="",
                ended_at=ended_at,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            error_message = str(exc) or "image task failed"
            ended_at = _now_iso()
            duration_ms = int((time.time() - started) * 1000)
            self._update_task(
                key,
                status=TASK_STATUS_ERROR,
                error=error_message,
                data=[],
                ended_at=ended_at,
                duration_ms=duration_ms,
            )

    def _log_call(
        self,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        suffix: str,
        *,
        request_preview: str = "",
        status: str = "success",
        error: str = "",
        urls: list[str] | None = None,
    ) -> None:
        endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
        summary_prefix = "图生图" if mode == "edit" else "文生图"
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": endpoint,
            "model": model,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if error:
            detail["error"] = error
        if urls:
            detail["urls"] = list(dict.fromkeys(urls))
        try:
            log_service.add(LOG_TYPE_CALL, f"{summary_prefix}{suffix}", detail)
        except Exception:
            pass

    def _update_task(self, key: str, **updates: Any) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return
            status = updates.pop("status", None)
            error = updates.pop("error", None)
            data = updates.pop("data", None)
            ended_at = updates.pop("ended_at", None)
            duration_ms = updates.pop("duration_ms", None)
            task.update(updates)
            if status is not None:
                self._set_task_status(
                    task,
                    str(status),
                    error="" if error is None else str(error),
                    data=data,
                    ended_at=ended_at,
                    duration_ms=duration_ms,
                )
            else:
                task["updated_at"] = _now_iso()
                self._sync_log_locked(task)
            self._save_locked()

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_items = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(raw_items, list):
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            status = _clean(item.get("status"))
            if status not in {
                TASK_STATUS_QUEUED,
                TASK_STATUS_RUNNING,
                TASK_STATUS_SUCCESS,
                TASK_STATUS_ERROR,
                TASK_STATUS_DELETED,
                TASK_STATUS_DELETED_RUNNING,
                TASK_STATUS_DELETED_SUCCESS,
                TASK_STATUS_DELETED_FAILED,
            }:
                status = TASK_STATUS_ERROR
            base_status = _clean(item.get("base_status"))
            if base_status not in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}:
                base_status = TASK_STATUS_ERROR if status in {TASK_STATUS_ERROR, TASK_STATUS_DELETED_FAILED} else TASK_STATUS_QUEUED
            task = {
                "id": task_id,
                "owner_id": owner,
                "key_id": _clean(item.get("key_id")),
                "key_name": _clean(item.get("key_name")),
                "role": _clean(item.get("role")),
                "status": status,
                "base_status": base_status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "size": _clean(item.get("size")),
                "endpoint": _clean(item.get("endpoint"), "/v1/images/edits" if item.get("mode") == "edit" else "/v1/images/generations"),
                "request_text": _clean(item.get("request_text")),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
                "started_at": _clean(item.get("started_at"), _clean(item.get("created_at"), _now_iso())),
                "results_deleted": bool(item.get("results_deleted")),
                "log_id": _clean(item.get("log_id")),
            }
            deleted_at = _clean(item.get("deleted_at"))
            if deleted_at:
                task["deleted_at"] = deleted_at
            data = item.get("data")
            if isinstance(data, list):
                task["data"] = data
                task["urls"] = _collect_image_urls(data)
            error = _clean(item.get("error"))
            if error:
                task["error"] = error
            ended_at = _clean(item.get("ended_at"))
            if ended_at:
                task["ended_at"] = ended_at
            duration_ms = item.get("duration_ms")
            if isinstance(duration_ms, int):
                task["duration_ms"] = duration_ms
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        items = sorted(self._tasks.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("base_status") in UNFINISHED_STATUSES or task.get("status") in UNFINISHED_STATUSES:
                task["ended_at"] = _now_iso()
                task["duration_ms"] = max(0, int((_timestamp(task.get("ended_at")) - _timestamp(task.get("started_at") or task.get("created_at"))) * 1000))
                self._set_task_status(task, TASK_STATUS_ERROR, error="服务已重启，未完成的图片任务已中断", data=task.get("data") if isinstance(task.get("data"), list) else [])
                changed = True
        return changed

    def _cleanup_locked(self) -> bool:
        try:
            retention_days = max(1, int(self.retention_days_getter()))
        except Exception:
            retention_days = 30
        cutoff = time.time() - retention_days * 86400
        removed_keys = [
            key
            for key, task in self._tasks.items()
            if task.get("status") in TERMINAL_STATUSES and _timestamp(task.get("updated_at")) < cutoff
        ]
        for key in removed_keys:
            self._tasks.pop(key, None)
        return bool(removed_keys)


image_task_service = ImageTaskService(DATA_DIR / "image_tasks.json")
