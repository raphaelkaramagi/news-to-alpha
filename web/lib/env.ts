/**
 * Flask API base URL (Railway / Render / local gunicorn). Used only on the server
 * by Route Handlers and RSC fetches — never expose as NEXT_PUBLIC_*.
 */
export function getApiBaseUrl(): string | undefined {
  const raw = process.env.API_BASE_URL?.trim();
  if (!raw) return undefined;
  return raw.replace(/\/$/, "");
}
