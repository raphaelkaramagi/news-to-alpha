"use client";
import { useState } from "react";
import { PricePredictionChart } from "@/components/charts/PricePredictionChart";
import { AccuracyTraceChart } from "@/components/charts/AccuracyTraceChart";
import { ChartWindowPicker } from "@/components/charts/ChartWindowPicker";
import { AccuracyPanel } from "@/components/ticker/AccuracyPanel";
import { VolatilityAccuracyPanel } from "@/components/ticker/VolatilityAccuracyPanel";
import { VolatilityTraceChart } from "@/components/charts/VolatilityTraceChart";
import { ResolvedStrip } from "@/components/ticker/ResolvedStrip";
import type { ChartWindow } from "@/lib/chartWindow";
import type { ModelId } from "@/lib/tickers";
import { MODEL_CHART_CONFIG, MODEL_DISPLAY_LABELS } from "@/lib/models";
import { shortDate } from "@/lib/forecastHorizon";

interface Props {
  ticker: string;
  selectedDate: string;
  targetDate?: string | null;
  model?: ModelId;
  window: ChartWindow;
  onWindowChange: (window: ChartWindow) => void;
}

export function PriceAccuracySection({
  ticker,
  selectedDate,
  targetDate,
  model = "ensemble",
  window,
  onWindowChange,
}: Props) {
  const chartCfg = MODEL_CHART_CONFIG[model];

  return (
    <section className="space-y-6 border rounded-lg p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-medium">
          Price & accuracy · {MODEL_DISPLAY_LABELS[model]}
        </h2>
        <ChartWindowPicker value={window} onChange={onWindowChange} />
      </div>

      <div>
        <p className="text-xs font-medium text-muted-foreground mb-1">Share price & forecast</p>
        <p className="text-[10px] text-muted-foreground mb-2">
          {chartCfg.probaLegend}
          {targetDate && (
            <>
              {" · "}
              Dashed line: close {shortDate(selectedDate)} → target {shortDate(targetDate)}
            </>
          )}
        </p>
        <PricePredictionChart
          ticker={ticker}
          selectedDate={selectedDate}
          targetDate={targetDate}
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

      <ResolvedStrip ticker={ticker} chartWindow={window} model={model} />

      <div>
        <p className="text-sm font-medium mb-3">Direction accuracy ({window}d window)</p>
        <AccuracyPanel ticker={ticker} window={window} model={model} />
      </div>

      <div className="pt-4 border-t space-y-6">
        <div>
          <p className="text-xs font-medium text-muted-foreground mb-1">
            Expected-move band accuracy
          </p>
          <p className="text-[10px] text-muted-foreground mb-2">
            Share of sessions where realized |return| stayed inside the ±% band
          </p>
          <VolatilityTraceChart
            ticker={ticker}
            window={window}
            selectedDate={selectedDate}
          />
        </div>

        <div>
          <p className="text-sm font-medium mb-3">Volatility ({window}d window)</p>
          <VolatilityAccuracyPanel ticker={ticker} window={window} />
        </div>
      </div>
    </section>
  );
}
