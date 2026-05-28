"use client";
import { useQuery } from "@tanstack/react-query";
import { TICKERS } from "@/lib/tickers";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

async function fetchDayAccuracy(date: string) {
  const results = await Promise.all(
    TICKERS.map(async (ticker) => {
      const res = await fetch(
        `/api/ticker?ticker=${ticker}&model=ensemble&date=${date}`,
        { cache: "no-store" }
      );
      if (!res.ok) return null;
      return (await res.json()) as TickerApiResponse;
    })
  );
  const rows = results.filter(Boolean) as TickerApiResponse[];
  const resolved = rows.filter((r) => r.actual_binary !== null);
  const hits = resolved.filter((r) => r.hit === 1).length;
  return { total: rows.length, resolved: resolved.length, hits };
}

interface Props {
  date: string | null;
}

export function DayAccuracyBanner({ date }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["day-accuracy", date],
    queryFn: () => fetchDayAccuracy(date!),
    enabled: !!date,
    staleTime: 60_000,
  });

  if (!date || isLoading) {
    return (
      <div className="h-10 rounded-lg bg-muted/50 animate-pulse" />
    );
  }

  if (!data) return null;

  const pending = data.total - data.resolved;

  if (data.resolved === 0) {
    return (
      <div className="rounded-lg border px-4 py-2.5 text-sm flex items-center justify-between gap-3">
        <span className="text-muted-foreground">
          <span className="font-mono text-xs">{date}</span> — results pending (market not closed yet)
        </span>
      </div>
    );
  }

  const pct = (data.hits / data.resolved) * 100;
  const good = pct >= 50;

  return (
    <div className="rounded-lg border px-4 py-2.5 text-sm flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
      <span className="text-muted-foreground">
        Ensemble on <span className="font-mono text-xs text-foreground">{date}</span>
      </span>
      <span className={cn("font-semibold tabular-nums", good ? "text-up" : "text-down")}>
        {data.hits}/{data.resolved} correct ({pct.toFixed(0)}%)
        {pending > 0 && (
          <span className="text-muted-foreground font-normal text-xs ml-2">
            · {pending} pending
          </span>
        )}
      </span>
    </div>
  );
}
