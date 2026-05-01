import { getApiBaseUrl } from "./env";

export function backendConfigured(): boolean {
  return Boolean(getApiBaseUrl());
}

/** Build absolute URL to Flask, e.g. path `/api/ticker` + query from incoming request. */
export function backendUrl(
  path: string,
  searchParams: URLSearchParams
): string | null {
  const base = getApiBaseUrl();
  if (!base) return null;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  const u = new URL(normalized, `${base}/`);
  searchParams.forEach((v, k) => {
    u.searchParams.set(k, v);
  });
  return u.toString();
}
