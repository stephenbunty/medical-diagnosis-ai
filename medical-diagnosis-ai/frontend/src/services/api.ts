import axios, { isAxiosError } from "axios";

/** Dev: Vite proxies /api → backend. Prod/Docker: set VITE_API_URL or use same-origin /api/v1. */
const baseURL = (import.meta.env.VITE_API_URL || "/api/v1").replace(/\/$/, "");

export const api = axios.create({
  baseURL,
  timeout: 120_000,
});

/** User-facing message from FastAPI `{ detail }` or network errors. */
export function apiErrorMessage(err: unknown, fallback: string): string {
  if (isAxiosError(err)) {
    if (!err.response) {
      return "Cannot reach the API. Is the backend running on port 8000?";
    }
    const detail = err.response.data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ") || fallback;
    }
  }
  return fallback;
}

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("mdai_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});
