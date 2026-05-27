"use client";
import type { ModelId } from "@/lib/tickers";
import { MODEL_DESCRIPTIONS } from "@/lib/models";

interface Props {
  model: ModelId;
}

export function ModelBlurb({ model }: Props) {
  const { title, body } = MODEL_DESCRIPTIONS[model];
  return (
    <div className="rounded-lg border bg-muted/30 px-4 py-3 text-sm">
      <p className="font-medium text-foreground">{title}</p>
      <p className="text-muted-foreground mt-1 leading-relaxed">{body}</p>
    </div>
  );
}
