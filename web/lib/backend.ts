import { NextResponse } from "next/server";
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

/**
 * Proxy to Flask; returns 503 JSON if the backend is down (ECONNREFUSED, timeout, etc.).
 */
export async function forwardToFlask(
  url: string,
  init?: RequestInit
): Promise<NextResponse> {
  const isMutation = Boolean(
    init?.method && !["GET", "HEAD"].includes(init.method)
  );
  const timeoutMs = isMutation ? 120_000 : 30_000;
  try {
    const res = await fetch(url, {
      ...init,
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
    const text = await res.text();
    return new NextResponse(text, {
      status: res.status,
      headers: {
        "content-type":
          res.headers.get("content-type") || "application/json; charset=utf-8",
      },
    });
  } catch {
    return NextResponse.json(
      {
        error:
          "Backend unreachable — start Flask at API_BASE_URL (e.g. gunicorn -b 127.0.0.1:8000 app.server:app).",
      },
      { status: 503 }
    );
  }
}
