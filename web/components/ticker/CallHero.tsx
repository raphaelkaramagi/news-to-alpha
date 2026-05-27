"use client";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { TICKER_TO_COMPANY } from "@/lib/tickers";
import { confidenceLabel } from "@/lib/confidence";

interface Props {
  data: TickerApiResponse;
  date: string;
  horizon?: number;
}

function formatReturn(pct: number): string {
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

export function CallHero({ data, date, horizon = 1 }: Props) {
  const company = TICKER_TO_COMPANY[data.ticker] ?? data.ticker;
  const isUp = data.binary === 1;
  const confidence = (data.confidence * 100).toFixed(0);
  const confLabel = confidenceLabel(data.confidence);
  const resolved = data.actual_binary !== null;
  const hit = data.hit;
  const moveLabel = horizon > 1 ? `${horizon}-day move` : "Next-day move";

  return (
    <div className="flex flex-col sm:flex-row sm:items-start gap-6">
      <div className="flex-1">
        <p className="text-sm text-muted-foreground mb-1">{company}</p>
        <div className="flex items-baseline gap-3">
          <h1 className="text-3xl font-semibold tracking-tight">{data.ticker}</h1>
          <span className="text-sm text-muted-foreground">{date}</span>
        </div>
      </div>

      <div className="flex items-start gap-3">
        {resolved && (
          <div className={cn(
            "flex flex-col items-start px-3 py-2 rounded-lg border text-sm min-w-[7rem]",
            hit === 1 ? "border-up/30 bg-up/5" : "border-down/30 bg-down/5"
          )}>
            <span className={cn("font-medium", hit === 1 ? "text-up" : "text-down")}>
              {hit === 1 ? "Correct" : "Wrong"}
            </span>
            {data.realized_return !== null && (
              <span className="text-xs text-muted-foreground mt-1">
                {moveLabel}:{" "}
                <span className={cn(
                  "font-medium tabular-nums",
                  data.realized_return >= 0 ? "text-up" : "text-down"
                )}>
                  {formatReturn(data.realized_return)}
                </span>
              </span>
            )}
          </div>
        )}

        <div className={cn(
          "flex flex-col items-center px-4 py-2 rounded-lg",
          isUp ? "bg-up/10 text-up border border-up/20" : "bg-down/10 text-down border border-down/20"
        )}>
          <span className="text-2xl font-bold tracking-tight">{isUp ? "UP" : "DOWN"}</span>
          <span className="text-xs text-muted-foreground mt-0.5">
            {confLabel} · {confidence}%
          </span>
        </div>
      </div>
    </div>
  );
}
