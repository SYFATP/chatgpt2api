"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { getValidatedAuthSession } from "@/lib/auth-session";
import { clearStoredAuthSession, type StoredAuthSession } from "@/store/auth";

type AuthContextValue = {
  isLoading: boolean;
  session: StoredAuthSession | null;
  logout: () => Promise<void>;
  refresh: () => Promise<StoredAuthSession | null>;
  setSession: (session: StoredAuthSession | null) => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSessionState] = useState<StoredAuthSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    try {
      const nextSession = await getValidatedAuthSession();
      setSessionState(nextSession);
      return nextSession;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const logout = useCallback(async () => {
    await clearStoredAuthSession();
    setSessionState(null);
  }, []);

  const setSession = useCallback((nextSession: StoredAuthSession | null) => {
    setSessionState(nextSession);
    setIsLoading(false);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      isLoading,
      session,
      logout,
      refresh,
      setSession,
    }),
    [isLoading, logout, refresh, session, setSession],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuthState() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuthState must be used within AuthProvider");
  }
  return context;
}
