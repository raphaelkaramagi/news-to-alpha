"use client";
import { CHART_WINDOWS, CHART_WINDOW_LABELS, type ChartWindow } from "@/lib/chartWindow";
import { cn } from "@/lib/utils";

interface Props {
  value: ChartWindow;
  onChange: (window: ChartWindow) => void;
}

export function ChartWindowPicker({ value, onChange }: Props) {
  return (
    <div className="flex gap-1">
      {CHART_WINDOWS.map((w) => (
        <button
          key={w}
          type="button"
          onClick={() => onChange(w)}
          className={cn(
            "px-2 py-0.5 rounded text-xs transition-colors",
            value === w
              ? "bg-primary text-primary-foreground"
              : "hover:bg-accent text-muted-foreground"
          )}
        >
          {CHART_WINDOW_LABELS[w]}
        </button>
      ))}
    </div>
  );
}
