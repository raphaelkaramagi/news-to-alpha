"use client";
import { ALLOWED_MODELS, type ModelId } from "@/lib/tickers";
import { MODEL_DISPLAY_LABELS } from "@/lib/models";
import type { PerModelEntry } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  selected: ModelId;
  onChange: (model: ModelId) => void;
  perModel: Record<string, PerModelEntry>;
}

export function ModelPicker({ selected, onChange, perModel }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {ALLOWED_MODELS.map((model) => {
        const entry = perModel[model];
        const isUp = entry ? entry.binary === 1 : null;
        return (
          <button
            key={model}
            onClick={() => onChange(model)}
            className={cn(
              "inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm border transition-colors",
              selected === model
                ? "bg-primary text-primary-foreground border-primary"
                : "bg-card border-border hover:bg-accent text-foreground"
            )}
          >
            {MODEL_DISPLAY_LABELS[model]}
            {entry && (
              <span className={cn(
                "text-xs font-medium",
                selected === model
                  ? "text-primary-foreground/70"
                  : isUp ? "text-up" : "text-down"
              )}>
                {(entry.proba * 100).toFixed(0)}%
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
