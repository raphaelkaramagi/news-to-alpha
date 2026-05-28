/** Recharts click/activeLabel → ISO date string for sitewide date nav. */
export function dateFromChartClick(
  state: { activeLabel?: string | number } | null | undefined
): string | null {
  const label = state?.activeLabel;
  if (label === undefined || label === null) return null;
  const s = String(label);
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
  return null;
}

export const CHART_CLICK_HINT = "Click a date on the chart to jump to that session.";
