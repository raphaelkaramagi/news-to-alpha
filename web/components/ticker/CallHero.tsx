"use client";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { TICKER_TO_COMPANY } from "@/lib/tickers";
import { confidenceLabel } from "@/lib/confidence";
import { formatReturnPct } from "@/lib/forecastHorizon";
import {
  CloseMoveVisual,
  ForecastDayPills,
} from "@/components/ticker/CloseMoveVisual";
import { useQuery } from "@tanstack/react-query";

interface Props {
  data: TickerApiResponse;
  date: string;
  horizon?: number;
}

async function fetchTickerPrices(
  ticker: string,
  date: string
): Promise<TickerApiResponse> {
  const res = await fetch(
    `/api/ticker?ticker=${ticker}&model=ensemble&date=${date}`,
    { cache: "no-store" }
  );
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

export function CallHero({ data, date, horizon = 1 }: Props) {
  const company = TICKER_TO_COMPANY[data.ticker] ?? data.ticker;
  const isUp = data.binary === 1;
  const confidence = (data.confidence * 100).toFixed(0);
  const confLabel = confidenceLabel(data.confidence);
  const resolved = data.actual_binary !== null;
  const hit = data.hit === 1;

  const { data: fresh } = useQuery({
    queryKey: ["ticker-prices", data.ticker, date],
    queryFn: () => fetchTickerPrices(data.ticker, date),
    enabled:
      !data.price_context?.start_close &&
      !data.price_context?.session_close,
    staleTime: 60_000,
  });

  const ctx = data.price_context ?? fresh?.price_context;
  const forecastDate =
    ctx?.end_close_date ??
    ctx?.target_date ??
    data.forecast_date ??
    fresh?.forecast_date;
  const returnPct = data.realized_return ?? ctx?.return_pct;
  const hasPrices = (ctx?.start_close ?? ctx?.session_close) != null;
  const expectedMove = data.expected_move_pct ?? fresh?.expected_move_pct;
  const forecastLow = data.forecast_low ?? fresh?.forecast_low;
  const forecastHigh = data.forecast_high ?? fresh?.forecast_high;

  return (
    <div className="flex flex-col lg:flex-row lg:items-start gap-4 lg:gap-6">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-muted-foreground mb-1">{company}</p>
        <h1 className="text-3xl font-semibold tracking-tight">{data.ticker}</h1>
        <ForecastDayPills dataDate={date} forecastDate={forecastDate} />
        {horizon > 1 && (
          <p className="text-xs text-muted-foreground mt-2">{horizon}-day horizon</p>
        )}
      </div>

      {/* Mobile: prices row, then call|result · Desktop: prices | call | result */}
      <div className="w-full lg:w-auto flex flex-col md:flex-row items-stretch rounded-xl border bg-card shadow-sm overflow-hidden shrink-0">
        <div
          className={cn(
            "flex items-center justify-center px-3 py-3 sm:px-4 bg-muted/20",
            "border-b md:border-b-0 md:border-r",
            !hasPrices && "opacity-60"
          )}
        >
          {hasPrices && ctx ? (
            <CloseMoveVisual
              ctx={ctx}
              showReturn={resolved}
              expectedMovePct={expectedMove}
              forecastLow={forecastLow}
              forecastHigh={forecastHigh}
            />
          ) : (
            <div className="text-center text-xs text-muted-foreground font-mono tabular-nums py-1">
              — → —
            </div>
          )}
        </div>

        <div className="flex flex-row md:contents min-w-0">
          <div
            className={cn(
              "flex flex-1 md:flex-none flex-col items-center justify-center px-4 py-3 sm:px-5 min-w-0 md:min-w-[5.5rem]",
              isUp ? "text-up bg-up/5" : "text-down bg-down/5",
              resolved && "border-r"
            )}
          >
            <span className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">
              {forecastDate ? "Next close" : "Call"}
            </span>
            <span className="text-2xl sm:text-3xl font-bold leading-none">
              {isUp ? "UP" : "DOWN"}
            </span>
            <span className="text-[10px] text-muted-foreground mt-1 text-center">
              {confLabel} · {confidence}%
            </span>
            {expectedMove != null && (
              <span className="text-[10px] text-muted-foreground mt-0.5 tabular-nums">
                ±{expectedMove.toFixed(1)}% expected move
              </span>
            )}
          </div>

          {resolved && (
            <div
              className={cn(
                "flex flex-1 md:flex-none flex-col items-center justify-center px-4 py-3 sm:px-5 min-w-0 md:min-w-[5rem] border-l md:border-l",
                hit ? "bg-up/10 text-up" : "bg-down/10 text-down"
              )}
            >
              <span className="text-[10px] uppercase tracking-wide opacity-70 mb-0.5">
                Result
              </span>
              <span className="text-xl sm:text-2xl font-bold leading-none">
                {hit ? "✓" : "✗"}
              </span>
              <span className="text-xs font-semibold mt-0.5">
                {hit ? "Correct" : "Wrong"}
              </span>
              {returnPct != null && (
                <span className="text-[10px] font-mono tabular-nums mt-1 opacity-80">
                  {formatReturnPct(returnPct)}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
