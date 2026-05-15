/** Build absolute URL for API-served images when frontend and API are on different hosts. */
export function assetUrl(path: string | null | undefined): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  const base = (import.meta.env.VITE_API_URL || "/api/v1").replace(/\/$/, "");
  if (base) return `${base}${path}`;
  return path;
}
