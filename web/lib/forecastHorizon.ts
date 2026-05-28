import type { PriceContext } from "./types";

/** Format a dollar price for display. */
export function formatPrice(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `$${value.toFixed(2)}`;
}

/** Format signed percent return. */
export function formatReturnPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

/** Short date for inline UI (MM-DD). */
export function shortDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.slice(5);
}

function startClose(ctx: PriceContext): number | null {
  return ctx.start_close ?? ctx.session_close;
}

function endClose(ctx: PriceContext): number | null {
  return ctx.end_close ?? ctx.target_close;
}

function startDate(ctx: PriceContext): string {
  return ctx.start_close_date ?? ctx.session_date;
}

function endDate(ctx: PriceContext): string | null {
  return ctx.end_close_date ?? ctx.target_date;
}

/** Plain-language rule matching the label generator. */
export function validationRuleLine(): string {
  return "Scored on close-to-close: selected day’s close → next trading day’s close (not same-day open→close).";
}

/** One-line explanation of what the selected session date means. */
export function forecastHorizonLine(ctx: PriceContext | null | undefined): string {
  if (!ctx) return validationRuleLine();
  const start = startDate(ctx);
  const end = endDate(ctx);
  if (!start || !end) return validationRuleLine();
  return `${start} forecast: will close on ${end} be above ${start}’s close?`;
}

/** Two-line close pair for badges. */
export function closePairLines(ctx: PriceContext | null | undefined): {
  startLabel: string;
  startPrice: string;
  endLabel: string;
  endPrice: string;
} | null {
  if (!ctx) return null;
  const sc = startClose(ctx);
  const ec = endClose(ctx);
  const sd = startDate(ctx);
  const ed = endDate(ctx);
  if (sc == null || !sd) return null;
  return {
    startLabel: `${shortDate(sd)} close`,
    startPrice: formatPrice(sc),
    endLabel: ctx.resolved && ed ? `${shortDate(ed)} close` : "next close",
    endPrice: ctx.resolved && ec != null ? formatPrice(ec) : "pending",
  };
}

/** Compact arrow line for market cards. */
export function forecastMoveLine(ctx: PriceContext | null | undefined): string | null {
  const pair = closePairLines(ctx);
  if (!pair) return null;
  const arrow = `${pair.startPrice} → ${pair.endPrice}`;
  if (ctx?.resolved && ctx.return_pct != null) {
    return `${arrow} (${formatReturnPct(ctx.return_pct)})`;
  }
  return arrow;
}
