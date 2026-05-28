"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  dates: string[];
  selected: string;
  onChange: (date: string) => void;
  className?: string;
  align?: "left" | "right";
}

function parseMonth(iso: string): { year: number; month: number } {
  const [y, m] = iso.split("-").map(Number);
  return { year: y, month: m - 1 };
}

export function PredictionDatePicker({
  dates,
  selected,
  onChange,
  className,
  align = "right",
}: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const dateSet = useMemo(() => new Set(dates), [dates]);
  const initial = parseMonth(selected || dates[dates.length - 1] || "2026-01-01");
  const [viewYear, setViewYear] = useState(initial.year);
  const [viewMonth, setViewMonth] = useState(initial.month);

  useEffect(() => {
    if (!open) return;
    const p = parseMonth(selected);
    setViewYear(p.year);
    setViewMonth(p.month);
  }, [open, selected]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const monthLabel = new Date(viewYear, viewMonth, 1).toLocaleString("en-US", {
    month: "long",
    year: "numeric",
  });

  const firstDow = new Date(viewYear, viewMonth, 1).getDay();
  const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();

  const cells: (string | null)[] = [];
  for (let i = 0; i < firstDow; i++) cells.push(null);
  for (let d = 1; d <= daysInMonth; d++) {
    const iso = `${viewYear}-${String(viewMonth + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
    cells.push(iso);
  }

  const prevMonth = () => {
    if (viewMonth === 0) {
      setViewYear((y) => y - 1);
      setViewMonth(11);
    } else setViewMonth((m) => m - 1);
  };
  const nextMonth = () => {
    if (viewMonth === 11) {
      setViewYear((y) => y + 1);
      setViewMonth(0);
    } else setViewMonth((m) => m + 1);
  };

  const pick = (d: string) => {
    onChange(d);
    setOpen(false);
  };

  return (
    <div ref={ref} className={cn("relative", className)}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1.5 font-mono text-xs tabular-nums min-w-[6rem] px-2 py-1 rounded hover:bg-accent transition-colors"
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        <Calendar size={13} className="text-muted-foreground shrink-0" />
        {selected}
      </button>

      {open && (
        <div
          className={cn(
            "absolute z-50 mt-1 w-[17rem] rounded-lg border bg-card shadow-lg p-3",
            align === "right" ? "right-0" : "left-0"
          )}
        >
          <div className="flex items-center justify-between mb-2">
            <button type="button" onClick={prevMonth} className="p-1 rounded hover:bg-accent" aria-label="Previous month">
              <ChevronLeft size={14} />
            </button>
            <span className="text-xs font-medium">{monthLabel}</span>
            <button type="button" onClick={nextMonth} className="p-1 rounded hover:bg-accent" aria-label="Next month">
              <ChevronRight size={14} />
            </button>
          </div>

          <div className="grid grid-cols-7 gap-0.5 text-[10px] text-muted-foreground text-center mb-1">
            {["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"].map((d) => (
              <span key={d}>{d}</span>
            ))}
          </div>

          <div className="grid grid-cols-7 gap-0.5">
            {cells.map((iso, i) => {
              if (!iso) return <span key={`e-${i}`} />;
              const available = dateSet.has(iso);
              const isSelected = iso === selected;
              return (
                <button
                  key={iso}
                  type="button"
                  disabled={!available}
                  onClick={() => available && pick(iso)}
                  className={cn(
                    "h-7 rounded text-xs tabular-nums transition-colors",
                    !available && "text-muted-foreground/25 cursor-not-allowed",
                    available && !isSelected && "bg-muted/60 hover:bg-accent font-medium",
                    isSelected && "bg-primary text-primary-foreground font-semibold"
                  )}
                >
                  {Number(iso.slice(8))}
                </button>
              );
            })}
          </div>

          <p className="text-[10px] text-muted-foreground mt-2 text-center">
            Shaded days have predictions
          </p>
        </div>
      )}
    </div>
  );
}
