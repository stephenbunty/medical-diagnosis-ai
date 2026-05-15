import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";

export type User = {
  id: number;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
};

type AuthCtx = {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName: string, role: string) => Promise<void>;
  logout: () => void;
  refreshMe: () => Promise<void>;
};

const AuthContext = createContext<AuthCtx | undefined>(undefined);

const TOKEN_KEY = "mdai_token";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
      void refreshMe();
    } else {
      localStorage.removeItem(TOKEN_KEY);
      setUser(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function refreshMe() {
    if (!token) return;
    try {
      const { data } = await api.get<User>("/auth/me");
      setUser(data);
    } catch {
      setToken(null);
      setUser(null);
    }
  }

  async function login(email: string, password: string) {
    const { data } = await api.post<{ access_token: string }>("/auth/login", { email, password });
    setToken(data.access_token);
  }

  async function register(email: string, password: string, fullName: string, role: string) {
    await api.post("/auth/register", { email, password, full_name: fullName, role });
    await login(email, password);
  }

  function logout() {
    setToken(null);
    setUser(null);
  }

  const value = useMemo(
    () => ({
      user,
      token,
      login,
      register,
      logout,
      refreshMe,
    }),
    [user, token],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
