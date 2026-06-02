"use client";
import type { PriceContext } from "@/lib/types";
import { cn } from "@/lib/utils";
import { formatPrice, formatReturnPct, shortDate } from "@/lib/forecastHorizon";
import { ArrowRight } from "lucide-react";

type Props = {
  ctx: PriceContext;
  compact?: boolean;
  showReturn?: boolean;
  className?: string;
  /** Expected move band (± pct) around start close when forecast unresolved. */
  forecastLow?: number | null;
  forecastHigh?: number | null;
  expectedMovePct?: number | null;
};

export function CloseMoveVisual({
  ctx,
  compact = false,
  showReturn = false,
  className,
  forecastLow,
  forecastHigh,
  expectedMovePct,
}: Props) {
  const start = ctx.start_close ?? ctx.session_close;
  const end = ctx.end_close ?? ctx.target_close;
  const startD = ctx.start_close_date ?? ctx.session_date;
  const endD = ctx.end_close_date ?? ctx.target_date;
  const resolved = ctx.resolved && end != null;

  if (start == null) return null;

  const ret = ctx.return_pct;
  const retUp = ret != null && ret >= 0;
  const showBand =
    !resolved &&
    expectedMovePct != null &&
    forecastLow != null &&
    forecastHigh != null &&
    start != null;

  return (
    <div className={cn("flex flex-col items-center gap-1", className)}>
      {showBand && (
        <p className="text-[10px] text-muted-foreground tabular-nums whitespace-nowrap">
          Expected move ±{expectedMovePct.toFixed(1)}%
          <span className="mx-1 text-muted-foreground/50">·</span>
          <span className="font-mono">
            {formatPrice(forecastLow)} – {formatPrice(forecastHigh)}
          </span>
        </p>
      )}
      <div
        className={cn(
          "flex items-center justify-center gap-1.5 sm:gap-2 max-w-full",
          compact ? "text-[10px]" : "text-xs"
        )}
      >
      <div className="text-center min-w-0 flex-1 sm:flex-none sm:min-w-[4rem]">
        <p className="text-muted-foreground uppercase tracking-wide text-[9px] truncate">
          {compact ? shortDate(startD) : `${shortDate(startD)} close`}
        </p>
        <p
          className={cn(
            "font-mono tabular-nums font-semibold truncate",
            !compact && "text-sm"
          )}
        >
          {formatPrice(start)}
        </p>
      </div>

      <div className="flex flex-col items-center gap-0.5 shrink-0 px-0.5">
        <ArrowRight
          className={cn("text-muted-foreground", compact ? "size-3" : "size-4")}
          aria-hidden
        />
        {showReturn && resolved && ret != null && (
          <span
            className={cn(
              "font-mono tabular-nums font-semibold text-[10px] sm:text-xs whitespace-nowrap",
              retUp ? "text-up" : "text-down"
            )}
          >
            {formatReturnPct(ret)}
          </span>
        )}
      </div>

      <div className="text-center min-w-0 flex-1 sm:flex-none sm:min-w-[4rem]">
        <p className="text-muted-foreground uppercase tracking-wide text-[9px] truncate">
          {resolved && endD
            ? compact
              ? shortDate(endD)
              : `${shortDate(endD)} close`
            : "next"}
        </p>
        <p
          className={cn(
            "font-mono tabular-nums font-semibold truncate",
            !compact && "text-sm"
          )}
        >
          {resolved ? formatPrice(end) : "—"}
        </p>
      </div>
      </div>
    </div>
  );
}
export function ForecastDayPills({
  dataDate,
  forecastDate,
}: {
  dataDate: string;
  forecastDate: string | null | undefined;
}) {
  return (
    <div className="mt-2">
      <span className="inline-flex items-center rounded-md border bg-muted/40 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
        Data · {shortDate(dataDate)}
      </span>
      <p className="text-xs text-muted-foreground mt-1.5">
        {forecastDate ? (
          <>
            This call is for{" "}
            <span className="font-mono text-foreground">{shortDate(forecastDate)}</span> close
          </>
        ) : (
          "This call is for the next session close"
        )}
      </p>
    </div>
  );
}
