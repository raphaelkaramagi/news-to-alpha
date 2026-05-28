"use client";
import { useState } from "react";
import { PricePredictionChart } from "@/components/charts/PricePredictionChart";
import { AccuracyTraceChart } from "@/components/charts/AccuracyTraceChart";
import { ChartWindowPicker } from "@/components/charts/ChartWindowPicker";
import { AccuracyPanel } from "@/components/ticker/AccuracyPanel";
import { ResolvedStrip } from "@/components/ticker/ResolvedStrip";
import type { ChartWindow } from "@/lib/chartWindow";
import type { ModelId } from "@/lib/tickers";
import { MODEL_CHART_CONFIG, MODEL_DISPLAY_LABELS } from "@/lib/models";

interface Props {
  ticker: string;
  selectedDate: string;
  model?: ModelId;
}

export function PriceAccuracySection({ ticker, selectedDate, model = "ensemble" }: Props) {
  const [window, setWindow] = useState<ChartWindow>("30");
  const chartCfg = MODEL_CHART_CONFIG[model];

  return (
    <section className="space-y-6 border rounded-lg p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-medium">
          Price & accuracy · {MODEL_DISPLAY_LABELS[model]}
        </h2>
        <ChartWindowPicker value={window} onChange={setWindow} />
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">Share price & forecast</p>
        <p className="text-[10px] text-muted-foreground mb-2">
          {chartCfg.probaLegend}
        </p>
        <PricePredictionChart
          ticker={ticker}
          selectedDate={selectedDate}
          model={model}
          window={window}
        />
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">Rolling hit rate</p>
        <p className="text-[10px] text-muted-foreground mb-2">
          {chartCfg.accuracyLegend}
        </p>
        <AccuracyTraceChart
          ticker={ticker}
          window={window}
          selectedDate={selectedDate}
          model={model}
        />
      </div>

      <ResolvedStrip ticker={ticker} window={window} model={model} />

      <div>
        <p className="text-sm font-medium mb-3">Accuracy ({window}d window)</p>
        <AccuracyPanel ticker={ticker} window={window} model={model} />
      </div>
    </section>
  );
}
