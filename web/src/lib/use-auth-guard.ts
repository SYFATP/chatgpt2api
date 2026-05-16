"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { useAuthState } from "@/components/auth-provider";
import {
  getDefaultRouteForRole,
  type AuthRole,
  type StoredAuthSession,
} from "@/store/auth";

type UseAuthGuardResult = {
  isCheckingAuth: boolean;
  session: StoredAuthSession | null;
};

export function useAuthGuard(allowedRoles?: AuthRole[]): UseAuthGuardResult {
  const router = useRouter();
  const { isLoading, session } = useAuthState();
  const allowedRolesKey = (allowedRoles || []).join(",");

  useEffect(() => {
    if (isLoading) {
      return;
    }

    const roleList = allowedRolesKey ? (allowedRolesKey.split(",") as AuthRole[]) : [];

    if (!session) {
      router.replace("/login");
      return;
    }

    if (roleList.length > 0 && !roleList.includes(session.role)) {
      router.replace(getDefaultRouteForRole(session.role));
    }
  }, [allowedRolesKey, isLoading, router, session]);

  return { isCheckingAuth: isLoading, session };
}

export function useRedirectIfAuthenticated() {
  const router = useRouter();
  const { isLoading, session } = useAuthState();

  useEffect(() => {
    if (isLoading || !session) {
      return;
    }
    router.replace(getDefaultRouteForRole(session.role));
  }, [isLoading, router, session]);

  return { isCheckingAuth: isLoading };
}
