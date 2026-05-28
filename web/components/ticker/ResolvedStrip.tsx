"use client";
import { useQuery } from "@tanstack/react-query";
import type { LastResolvedRow } from "@/lib/types";
import { cn } from "@/lib/utils";
import { chartWindowDays, type ChartWindow } from "@/lib/chartWindow";

interface Props {
  ticker: string;
  window: ChartWindow;
}

async function fetchLastResolved(
  ticker: string,
  n: number
): Promise<{ ticker: string; rows: LastResolvedRow[] }> {
  const res = await fetch(`/api/last-resolved?ticker=${ticker}&n=${n}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

export function ResolvedStrip({ ticker, window }: Props) {
  const n = chartWindowDays(window);

  const { data } = useQuery({
    queryKey: ["last-resolved", ticker, n],
    queryFn: () => fetchLastResolved(ticker, n),
    staleTime: 300_000,
  });

  if (!data?.rows?.length) return null;

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">Last {n} resolved</p>
      <div className="flex flex-wrap gap-1.5">
        {data.rows.map((row, i) => (
          <div
            key={`${row.date}-${i}`}
            title={`${row.date}: ${row.hit === 1 ? "Correct" : "Wrong"}`}
            className={cn(
              "size-5 rounded-sm flex items-center justify-center text-xs font-medium",
              row.hit === 1
                ? "bg-up/15 text-up"
                : row.hit === 0
                ? "bg-down/15 text-down"
                : "bg-muted text-muted-foreground"
            )}
          >
            {row.hit === 1 ? "✓" : row.hit === 0 ? "✗" : "·"}
          </div>
        ))}
      </div>
    </div>
  );
}
