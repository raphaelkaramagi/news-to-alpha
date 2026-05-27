"use client";
import { useQuery } from "@tanstack/react-query";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

async function fetchAllDates(): Promise<string[]> {
  const res = await fetch("/api/dates?ticker=AAPL", { cache: "no-store" });
  if (!res.ok) return [];
  const data = await res.json();
  return data.dates ?? [];
}

export function GlobalDateNav() {
  const { selectedDate, latestDate, setSelectedDate, goLatest } = useSelectedDate();
  const { data: dates = [] } = useQuery({
    queryKey: ["global-dates"],
    queryFn: fetchAllDates,
    staleTime: 120_000,
  });

  if (!selectedDate || dates.length === 0) return null;

  const idx = dates.indexOf(selectedDate);
  const goPrev = () => { if (idx > 0) setSelectedDate(dates[idx - 1]); };
  const goNext = () => { if (idx < dates.length - 1) setSelectedDate(dates[idx + 1]); };
  const isLatest = selectedDate === latestDate;

  return (
    <div className="flex items-center gap-1.5 text-xs">
      <button
        onClick={goPrev}
        disabled={idx <= 0}
        className="p-1 rounded hover:bg-accent disabled:opacity-30"
        aria-label="Previous date"
      >
        <ChevronLeft size={14} />
      </button>
      <span className="font-mono tabular-nums min-w-[5.5rem] text-center text-muted-foreground">
        {selectedDate}
      </span>
      <button
        onClick={goNext}
        disabled={isLatest || idx >= dates.length - 1}
        className="p-1 rounded hover:bg-accent disabled:opacity-30"
        aria-label="Next date"
      >
        <ChevronRight size={14} />
      </button>
      {!isLatest && latestDate && (
        <button
          onClick={goLatest}
          className={cn(
            "ml-1 px-2 py-0.5 rounded text-[10px] hidden sm:inline",
            "text-muted-foreground hover:text-foreground hover:bg-accent"
          )}
        >
          Latest
        </button>
      )}
    </div>
  );
}
