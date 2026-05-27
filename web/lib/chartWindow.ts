export const CHART_WINDOWS = ["7", "30", "90"] as const;
export type ChartWindow = (typeof CHART_WINDOWS)[number];

export const CHART_WINDOW_LABELS: Record<ChartWindow, string> = {
  "7": "7d",
  "30": "30d",
  "90": "90d",
};

export function chartWindowDays(window: ChartWindow): number {
  return parseInt(window, 10);
}
