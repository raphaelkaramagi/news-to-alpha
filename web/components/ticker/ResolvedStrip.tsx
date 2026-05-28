"use client";
import { useQuery } from "@tanstack/react-query";
import { usePathname } from "next/navigation";
import type { ModelId } from "@/lib/tickers";
import { cn } from "@/lib/utils";
import { OutcomeMark } from "@/components/shared/OutcomeMark";
import { type ChartWindow } from "@/lib/chartWindow";
import { formatReturnPct, shortDate } from "@/lib/forecastHorizon";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";

interface Props {
  ticker: string;
  chartWindow: ChartWindow;
  model?: ModelId;
}

type SummaryRow = {
  date: string;
  hit: number;
  return?: number | null;
};

type SummaryResponse = {
  n: number;
  hits: number;
  rows: SummaryRow[];
};

async function fetchResolvedWindow(
  ticker: string,
  window: ChartWindow,
  model: ModelId
): Promise<SummaryResponse> {
  const params = new URLSearchParams({ ticker, window, model });
  const res = await fetch(`/api/accuracy-summary?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  const data = await res.json();
  return {
    n: data.n ?? 0,
    hits: data.hits ?? 0,
    rows: (data.rows ?? []) as SummaryRow[],
  };
}

function rowTitle(row: SummaryRow): string {
  const parts = [`${row.date}: ${row.hit === 1 ? "Correct" : "Wrong"}`];
  if (row.return != null) parts.push(formatReturnPct(row.return));
  return parts.join(" · ");
}

function ResolvedDot({
  row,
  selected,
  onSelect,
}: {
  row: SummaryRow;
  selected: boolean;
  onSelect: (date: string) => void;
}) {
  const variant = row.hit === 1 ? "correct" : row.hit === 0 ? "wrong" : "pending";

  return (
    <button
      type="button"
      title={rowTitle(row)}
      aria-label={`${row.date}: ${row.hit === 1 ? "correct" : "wrong"}`}
      aria-current={selected ? "true" : undefined}
      onClick={() => onSelect(row.date)}
      className={cn(
        "shrink-0 rounded-sm transition-opacity focus-visible:outline-none focus-visible:opacity-80",
        selected ? "opacity-100" : "opacity-90 hover:opacity-100"
      )}
    >
      <OutcomeMark variant={variant} size="md" />
    </button>
  );
}

/** Same last-N resolved sessions as Accuracy panel / chart window picker. */
export function ResolvedStrip({ ticker, chartWindow, model = "ensemble" }: Props) {
  const { selectedDate, setSelectedDate } = useSelectedDate();
  const pathname = usePathname();

  const handleSelect = (date: string) => {
    setSelectedDate(date);
    if (pathname?.startsWith("/t/")) {
      const url = `${pathname}?date=${date}`;
      globalThis.history.replaceState(null, "", url);
    }
  };

  const { data } = useQuery({
    queryKey: ["accuracy-summary", ticker, chartWindow, model, "strip"],
    queryFn: () => fetchResolvedWindow(ticker, chartWindow, model),
    staleTime: 300_000,
  });

  if (!data?.rows?.length) return null;

  return (
    <div>
      <p className="text-xs text-muted-foreground mb-2">
        Last {chartWindow}d resolved
        <span className="ml-1.5 tabular-nums">
          · {data.hits}/{data.n}
        </span>
      </p>

      <div className="flex flex-wrap gap-1">
        {data.rows.map((row) => (
          <ResolvedDot
            key={row.date}
            row={row}
            selected={selectedDate === row.date}
            onSelect={handleSelect}
          />
        ))}
      </div>

      <p className="text-[10px] text-muted-foreground mt-1.5">
        Click a mark to jump to that session · oldest {shortDate(data.rows[0]?.date)} → newest{" "}
        {shortDate(data.rows[data.rows.length - 1]?.date)}
      </p>
    </div>
  );
}
