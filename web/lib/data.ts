import { getApiBaseUrl } from "./env";
import type { DataStatus, TickerApiResponse } from "./types";

export async function fetchTickerServer(
  ticker: string,
  model: string
): Promise<{ ok: boolean; data: TickerApiResponse | null; status: number; apiError?: string }> {
  const base = getApiBaseUrl();
  if (!base) {
    return { ok: false, data: null, status: 503 };
  }
  const url = new URL("/api/ticker", base);
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("model", model);
  const res = await fetch(url.toString(), { cache: "no-store" });
  const raw = await res.json().catch(() => null);
  if (!res.ok) {
    const msg =
      raw && typeof raw === "object" && "error" in raw && typeof (raw as { error: unknown }).error === "string"
        ? (raw as { error: string }).error
        : undefined;
    return { ok: false, data: null, status: res.status, apiError: msg };
  }
  const data = raw as TickerApiResponse;
  return { ok: true, data, status: 200 };
}

export async function fetchDataStatusServer(): Promise<DataStatus | null> {
  const base = getApiBaseUrl();
  if (!base) return null;
  const res = await fetch(`${base}/api/data-status`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json() as Promise<DataStatus>;
}