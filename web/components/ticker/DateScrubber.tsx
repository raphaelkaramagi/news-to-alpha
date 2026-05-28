"use client";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { PredictionDatePicker } from "@/components/shared/PredictionDatePicker";

interface Props {
  ticker: string;
  selected: string;
  onChange: (date: string) => void;
}

async function fetchDates(ticker: string): Promise<{ dates: string[] }> {
  const res = await fetch(`/api/dates?ticker=${ticker}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

export function DateScrubber({ ticker, selected, onChange }: Props) {
  const { data } = useQuery({
    queryKey: ["dates", ticker],
    queryFn: () => fetchDates(ticker),
    staleTime: 300_000,
  });

  const dates = data?.dates ?? [];
  const idx = dates.indexOf(selected);

  const goPrev = () => { if (idx > 0) onChange(dates[idx - 1]); };
  const goNext = () => { if (idx < dates.length - 1) onChange(dates[idx + 1]); };
  const goLatest = () => { if (dates.length) onChange(dates[dates.length - 1]); };

  if (!dates.length) return null;

  const isLatest = idx === dates.length - 1;
  const gapHint =
    idx > 0 && idx < dates.length
      ? (() => {
          const prev = new Date(`${dates[idx - 1]}T12:00:00`);
          const cur = new Date(`${dates[idx]}T12:00:00`);
          const gapDays = Math.round((cur.getTime() - prev.getTime()) / 86400000);
          return gapDays > 1
            ? `${gapDays - 1} non-trading day(s) between sessions`
            : undefined;
        })()
      : undefined;

  return (
    <div className="flex flex-col items-end gap-0.5">
      <div className="flex items-center gap-2 text-sm">
        <button
          onClick={goPrev}
          disabled={idx <= 0}
          className="p-1 rounded hover:bg-accent disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          aria-label="Previous date"
        >
          <ChevronLeft size={16} />
        </button>
        <PredictionDatePicker
          dates={dates}
          selected={selected}
          onChange={onChange}
          align="right"
        />
        <button
          onClick={goNext}
          disabled={isLatest}
          className="p-1 rounded hover:bg-accent disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          aria-label="Next date"
        >
          <ChevronRight size={16} />
        </button>
        {!isLatest && (
          <button
            onClick={goLatest}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors ml-1"
          >
            Latest →
          </button>
        )}
      </div>
      {gapHint && (
        <p className="text-[10px] text-muted-foreground">{gapHint}</p>
      )}
    </div>
  );
}
