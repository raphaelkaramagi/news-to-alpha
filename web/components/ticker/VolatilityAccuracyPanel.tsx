"use client";
import { useQuery } from "@tanstack/react-query";
import type { ChartWindow } from "@/lib/chartWindow";
import { fetchVolatilitySummary } from "@/lib/volatilityAccuracy";

interface Props {
  ticker: string;
  window: ChartWindow;
}

export function VolatilityAccuracyPanel({ ticker, window }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["volatility-summary", ticker, window],
    queryFn: () => fetchVolatilitySummary(ticker, window),
    staleTime: 300_000,
  });

  if (isLoading) {
    return <div className="h-16 rounded-lg bg-muted animate-pulse" />;
  }

  if (!data || data.n === 0) {
    return <p className="text-sm text-muted-foreground">No resolved volatility data</p>;
  }

  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-3">
        <span className="text-4xl font-bold tabular-nums">
          {data.accuracy !== null ? `${(data.accuracy * 100).toFixed(0)}%` : "—"}
        </span>
        <span className="text-sm text-muted-foreground">
          {data.hits}/{data.n} within ± band
        </span>
      </div>
      {data.mae_pct != null && (
        <p className="text-xs text-muted-foreground tabular-nums">
          Avg calibration error {data.mae_pct.toFixed(2)}% (|actual − predicted|)
        </p>
      )}
    </div>
  );
}
