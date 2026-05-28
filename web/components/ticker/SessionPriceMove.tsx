"use client";
import type { PriceContext } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  closePairLines,
  forecastHorizonLine,
  formatReturnPct,
  validationRuleLine,
} from "@/lib/forecastHorizon";

type Props = {
  ctx: PriceContext | null | undefined;
  compact?: boolean;
  className?: string;
};

/** Close-to-close prices used for scoring (matches labels.close_t / close_t_plus_1). */
export function SessionPriceMove({ ctx, compact = false, className }: Props) {
  if (!ctx?.session_date) return null;

  const pair = closePairLines(ctx);
  if (!pair) return null;

  const ret = ctx.return_pct;
  const retUp = ret != null && ret >= 0;

  if (compact) {
    const line = `${pair.startPrice} → ${pair.endPrice}`;
    return (
      <p
        className={cn(
          "text-[10px] text-muted-foreground font-mono tabular-nums truncate",
          className
        )}
      >
        {line}
        {ctx.resolved && ret != null && (
          <span className={cn("ml-1", retUp ? "text-up" : "text-down")}>
            ({formatReturnPct(ret)})
          </span>
        )}
      </p>
    );
  }

  return (
    <div className={cn("rounded-lg border bg-muted/20 px-4 py-3 space-y-2", className)}>
      <p className="text-xs text-muted-foreground">{forecastHorizonLine(ctx)}</p>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 font-mono tabular-nums text-sm">
        <span>
          <span className="text-[10px] uppercase text-muted-foreground mr-1">
            {pair.startLabel}
          </span>
          {pair.startPrice}
        </span>
        <span className="text-muted-foreground">→</span>
        <span>
          <span className="text-[10px] uppercase text-muted-foreground mr-1">
            {pair.endLabel}
          </span>
          {pair.endPrice}
        </span>
        {ctx.resolved && ret != null && (
          <span className={cn("text-sm font-semibold", retUp ? "text-up" : "text-down")}>
            {formatReturnPct(ret)}
          </span>
        )}
      </div>
      <p className="text-[10px] text-muted-foreground">{validationRuleLine()}</p>
    </div>
  );
}
