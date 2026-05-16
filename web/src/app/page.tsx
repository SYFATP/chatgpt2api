"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { useAuthState } from "@/components/auth-provider";
import { getDefaultRouteForRole } from "@/store/auth";

export default function HomePage() {
  const router = useRouter();
  const { isLoading, session } = useAuthState();

  useEffect(() => {
    if (isLoading) {
      return;
    }
    router.replace(session ? getDefaultRouteForRole(session.role) : "/login");
  }, [isLoading, router, session]);

  return null;
}
