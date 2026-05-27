import type { TickerApiResponse } from "./types";

/** True when model directions in `per_model` are not all identical. */
export function hasMixedSignals(data: TickerApiResponse): boolean {
  const bins = Object.values(data.per_model).map((m) => m.binary);
  if (bins.length < 2) return false;
  const first = bins[0];
  return bins.some((b) => b !== first);
}
