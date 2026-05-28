"use client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { TICKERS, TICKER_TO_COMPANY } from "@/lib/tickers";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { formatReturnPct } from "@/lib/forecastHorizon";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { MarketsOverview } from "@/components/markets/MarketsOverview";
import { DayAccuracyBanner } from "@/components/markets/DayAccuracyBanner";
import { OutcomeDot, OutcomeLegend } from "@/components/markets/OutcomeDot";

async function fetchTicker(symbol: string, date: string | null): Promise<TickerApiResponse> {
  const params = new URLSearchParams({ ticker: symbol, model: "ensemble" });
  if (date) params.set("date", date);
  const res = await fetch(`/api/ticker?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

function TickerCard({ symbol, date }: { symbol: string; date: string | null }) {
  const { data, isLoading } = useQuery({
    queryKey: ["ticker", symbol, date],
    queryFn: () => fetchTicker(symbol, date),
    staleTime: 60_000,
  });

  const company = TICKER_TO_COMPANY[symbol] ?? symbol;
  const href = date ? `/t/${symbol}?date=${date}` : `/t/${symbol}`;
  const resolved = data?.actual_binary !== null;
  const movePct = data?.realized_return ?? data?.price_context?.return_pct;

  return (
    <Link
      href={href}
      className="group flex items-center justify-between gap-3 p-3 rounded-lg border bg-card hover:bg-accent transition-colors min-h-[3.25rem]"
    >
      <div className="flex items-center gap-2.5 min-w-0">
        <OutcomeDot data={isLoading ? undefined : data} size="md" />
        <div className="min-w-0 leading-tight">
          <p className="font-medium text-sm">{symbol}</p>
          <p className="text-[11px] text-muted-foreground truncate">{company}</p>
        </div>
      </div>

      <div className="text-right shrink-0 leading-tight">
        {isLoading ? (
          <span className="text-xs text-muted-foreground/50">—</span>
        ) : data ? (
          <>
            <p
              className={cn(
                "text-sm font-semibold tabular-nums",
                data.binary === 1 ? "text-up" : "text-down"
              )}
            >
              {data.binary === 1 ? "UP" : "DOWN"}
              <span className="text-muted-foreground font-normal text-xs ml-1">
                {(data.confidence * 100).toFixed(0)}%
              </span>
            </p>
            {resolved && movePct != null && (
              <p
                className={cn(
                  "text-[11px] font-mono tabular-nums mt-0.5",
                  movePct >= 0 ? "text-up/80" : "text-down/80"
                )}
              >
                {formatReturnPct(movePct)} actual
              </p>
            )}
          </>
        ) : (
          <span className="text-xs text-muted-foreground">—</span>
        )}
      </div>
    </Link>
  );
}

export function TickerGrid() {
  const { selectedDate } = useSelectedDate();

  return (
    <div className="space-y-8">
      <DayAccuracyBanner date={selectedDate} />
      <div className="space-y-3">
        <OutcomeLegend />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {TICKERS.map((symbol) => (
            <TickerCard key={symbol} symbol={symbol} date={selectedDate} />
          ))}
        </div>
      </div>
      <MarketsOverview />
    </div>
  );
}
