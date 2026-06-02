import type { ChartWindow } from "./chartWindow";

export type VolatilityAccuracySummary = {
  scope?: string;
  window: string;
  n: number;
  hits: number;
  accuracy: number | null;
  mae_pct: number | null;
};

export async function fetchVolatilitySummary(
  ticker: string,
  window: ChartWindow
): Promise<VolatilityAccuracySummary> {
  const params = new URLSearchParams({ ticker, window });
  const res = await fetch(`/api/volatility-summary?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("volatility-summary failed");
  return res.json();
}

export type VolatilityTracePoint = { date: string; accuracy: number | null };

export async function fetchVolatilityTrace(
  ticker: string | undefined,
  rollingWindow: number,
  displayDays: number
): Promise<VolatilityTracePoint[]> {
  const params = new URLSearchParams({ window: String(rollingWindow) });
  if (ticker) params.set("ticker", ticker);
  const res = await fetch(`/api/volatility-trace?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("volatility-trace failed");
  const data = await res.json();
  const series: VolatilityTracePoint[] = data.series ?? [];
  return series.filter((p) => p.accuracy !== null).slice(-displayDays);
}
