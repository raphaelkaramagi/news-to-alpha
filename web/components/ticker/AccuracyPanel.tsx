"use client";
import { useQuery } from "@tanstack/react-query";
import type { AccuracySummary } from "@/lib/types";
import type { ChartWindow } from "@/lib/chartWindow";
import type { ModelId } from "@/lib/tickers";

interface Props {
  ticker: string;
  window: ChartWindow;
  model?: ModelId;
}

async function fetchAccuracy(
  ticker: string,
  window: string,
  model: ModelId
): Promise<AccuracySummary> {
  const params = new URLSearchParams({ ticker, window, model });
  const res = await fetch(`/api/accuracy-summary?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

export function AccuracyPanel({ ticker, window, model = "ensemble" }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["accuracy-summary", ticker, window, model],
    queryFn: () => fetchAccuracy(ticker, window, model),
    staleTime: 300_000,
  });

  if (isLoading) {
    return <div className="h-16 rounded-lg bg-muted animate-pulse" />;
  }

  if (!data) {
    return <p className="text-sm text-muted-foreground">No data</p>;
  }

  return (
    <div className="flex items-baseline gap-3">
      <span className="text-4xl font-bold tabular-nums">
        {data.accuracy !== null ? `${(data.accuracy * 100).toFixed(0)}%` : "—"}
      </span>
      <span className="text-sm text-muted-foreground">
        {data.hits}/{data.n} correct
      </span>
    </div>
  );
}
