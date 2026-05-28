"use client";
import { useQuery } from "@tanstack/react-query";
import { TICKERS } from "@/lib/tickers";
import type { TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { shortDate } from "@/lib/forecastHorizon";

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
  const forecastDate =
    rows.find((r) => r.forecast_date)?.forecast_date ??
    rows.find((r) => r.price_context?.target_date)?.price_context?.target_date ??
    null;
  return { total: rows.length, resolved: resolved.length, hits, forecastDate };
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
    return <div className="h-10 rounded-lg bg-muted/50 animate-pulse" />;
  }

  if (!data) return null;

  const pending = data.total - data.resolved;
  const forecastLabel = data.forecastDate
    ? shortDate(data.forecastDate)
    : "next session";

  if (data.resolved === 0) {
    return (
      <div className="rounded-lg border px-4 py-2.5 text-sm flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="text-muted-foreground text-xs">
          <span className="font-mono">{date}</span>
          <span className="mx-1.5">·</span>
          calls for <span className="font-mono text-foreground">{forecastLabel}</span> close
          <span className="mx-1.5">·</span>
          pending
        </span>
      </div>
    );
  }

  const pct = (data.hits / data.resolved) * 100;
  const good = pct >= 50;

  return (
    <div className="rounded-lg border px-4 py-2.5 text-sm flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
      <span className="text-muted-foreground text-xs">
        <span className="font-mono">{date}</span>
        <span className="mx-1.5">·</span>
        calls for <span className="font-mono text-foreground">{forecastLabel}</span> close
      </span>
      <span className={cn("font-semibold tabular-nums text-sm", good ? "text-up" : "text-down")}>
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
