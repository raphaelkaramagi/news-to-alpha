"use client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { TICKERS, TICKER_TO_COMPANY } from "@/lib/tickers";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { MarketsOverview } from "@/components/markets/MarketsOverview";
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

  return (
    <Link
      href={href}
      className="group flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent transition-colors"
    >
      <div className="flex items-center gap-3 min-w-0">
        <OutcomeDot data={isLoading ? undefined : data} size={10} />
        <div className="min-w-0">
          <p className="font-medium text-sm leading-none">{symbol}</p>
          <p className="text-xs text-muted-foreground truncate mt-0.5">{company}</p>
        </div>
      </div>
      {isLoading ? (
        <span className="text-xs text-muted-foreground/50">—</span>
      ) : data ? (
        <span className={cn(
          "text-xs font-medium tabular-nums",
          data.binary === 1 ? "text-up" : "text-down"
        )}>
          {data.binary === 1 ? "UP" : "DOWN"}
          <span className="text-muted-foreground font-normal ml-1">
            {(data.confidence * 100).toFixed(0)}%
          </span>
        </span>
      ) : (
        <span className="text-xs text-muted-foreground">—</span>
      )}
    </Link>
  );
}

export function TickerGrid() {
  const { selectedDate } = useSelectedDate();

  return (
    <div className="space-y-8">
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
