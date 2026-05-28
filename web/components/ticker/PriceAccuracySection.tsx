"use client";
import { useState } from "react";
import { PricePredictionChart } from "@/components/charts/PricePredictionChart";
import { AccuracyTraceChart } from "@/components/charts/AccuracyTraceChart";
import { ChartWindowPicker } from "@/components/charts/ChartWindowPicker";
import { AccuracyPanel } from "@/components/ticker/AccuracyPanel";
import { ResolvedStrip } from "@/components/ticker/ResolvedStrip";
import type { ChartWindow } from "@/lib/chartWindow";
import type { ModelId } from "@/lib/tickers";

interface Props {
  ticker: string;
  selectedDate: string;
  model?: ModelId;
}

export function PriceAccuracySection({ ticker, selectedDate, model = "ensemble" }: Props) {
  const [window, setWindow] = useState<ChartWindow>("30");

  return (
    <section className="space-y-6 border rounded-lg p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-medium">Price & accuracy</h2>
        <ChartWindowPicker value={window} onChange={setWindow} />
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">Share price & forecast</p>
        <p className="text-[10px] text-muted-foreground mb-2">
          Black = close · green = model&apos;s UP probability (right axis)
        </p>
        <PricePredictionChart
          key={`${ticker}-${model}`}
          ticker={ticker}
          selectedDate={selectedDate}
          model={model}
          window={window}
        />
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">Rolling hit rate</p>
        <p className="text-[10px] text-muted-foreground mb-2">
          % of recent sessions the ensemble got right · dashed = 50% (coin flip)
        </p>
        <AccuracyTraceChart
          ticker={ticker}
          window={window}
          selectedDate={selectedDate}
        />
      </div>

      <ResolvedStrip ticker={ticker} window={window} />

      <div>
        <p className="text-sm font-medium mb-3">Accuracy ({window}d window)</p>
        <AccuracyPanel ticker={ticker} window={window} />
      </div>
    </section>
  );
}
