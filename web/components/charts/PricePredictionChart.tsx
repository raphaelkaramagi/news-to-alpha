"use client";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Area,
} from "recharts";
import type { HistoryResponse } from "@/lib/types";
import type { ChartWindow } from "@/lib/chartWindow";
import { chartWindowDays } from "@/lib/chartWindow";
import type { ModelId } from "@/lib/tickers";
import { MODEL_CHART_CONFIG } from "@/lib/models";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { CHART_CLICK_HINT, dateFromChartClick } from "@/lib/chartClick";

interface Props {
  ticker: string;
  selectedDate: string;
  model?: ModelId;
  window: ChartWindow;
}

async function fetchHistory(ticker: string, window = 90): Promise<HistoryResponse> {
  const res = await fetch(`/api/history?ticker=${ticker}&window=${window}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("fetch failed");
  const text = await res.text();
  const safe = text.replace(/:\s*NaN\b/g, ": null").replace(/:\s*-NaN\b/g, ": null");
  return JSON.parse(safe) as HistoryResponse;
}

const MODEL_PROBA_COL: Record<ModelId, string> = {
  ensemble: "ensemble_pred_proba",
  lstm: "financial_pred_proba",
  tfidf: "news_tfidf_pred_proba",
  embeddings: "news_embeddings_pred_proba",
};

function probaDomain(values: number[]): [number, number] {
  if (values.length === 0) return [0, 1];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  // Zoom axis when model outputs cluster (news models often ~0.45–0.55)
  if (span < 0.15) {
    const pad = Math.max(0.05, span * 0.5);
    return [Math.max(0, min - pad), Math.min(1, max + pad)];
  }
  return [0, 1];
}

export function PricePredictionChart({
  ticker,
  selectedDate,
  model = "ensemble",
  window,
}: Props) {
  const days = chartWindowDays(window);
  const { setSelectedDate } = useSelectedDate();
  const chartCfg = MODEL_CHART_CONFIG[model];
  const probaCol = MODEL_PROBA_COL[model];

  const { data, isLoading, isError } = useQuery({
    queryKey: ["history", ticker, days],
    queryFn: () => fetchHistory(ticker, days),
    staleTime: 60_000,
  });

  const chartData = useMemo(() => {
    if (!data) return [];
    const priceMap = new Map(data.prices.map((p) => [p.date, p.close]));
    const predMap = new Map(data.predictions.map((p) => [p.prediction_date, p]));
    const allDates = Array.from(
      new Set([
        ...data.prices.map((p) => p.date),
        ...data.predictions.map((p) => p.prediction_date),
      ])
    ).sort();
    return allDates
      .map((date) => {
        const pred = predMap.get(date);
        const raw = pred
          ? (pred as unknown as Record<string, number | null>)[probaCol]
          : null;
        return {
          date,
          close: priceMap.get(date) ?? null,
          proba: raw != null && !Number.isNaN(raw) ? raw : null,
        };
      })
      .slice(-days);
  }, [data, days, probaCol]);

  if (isLoading) {
    return <div className="h-60 w-full rounded-lg bg-muted animate-pulse" />;
  }

  if (isError || !data) {
    return (
      <p className="text-sm text-muted-foreground py-6 text-center">
        Could not load price history for {ticker}.
      </p>
    );
  }

  const prices = chartData.map((d) => d.close).filter((v): v is number => v !== null);
  const probaVals = chartData.map((d) => d.proba).filter((v): v is number => v !== null);

  if (chartData.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-6 text-center">
        No chart data for {ticker} yet.
      </p>
    );
  }

  const minPrice = prices.length ? Math.min(...prices) * 0.998 : 0;
  const maxPrice = prices.length ? Math.max(...prices) * 1.002 : 100;
  const hasPrices = prices.length > 0;
  const hasProba = probaVals.length > 0;
  const [probaMin, probaMax] = probaDomain(probaVals);

  return (
    <div className="w-full h-60 min-h-[240px]">
      <ResponsiveContainer width="100%" height={240}>
        <ComposedChart
          data={chartData}
          margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
          onClick={(state) => {
            const d = dateFromChartClick(state);
            if (d) setSelectedDate(d);
          }}
          style={{ cursor: "pointer" }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.5} />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            tickFormatter={(v) => String(v).slice(5)}
            interval="preserveStartEnd"
          />
          {hasPrices && (
            <YAxis
              yAxisId="price"
              domain={[minPrice, maxPrice]}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              tickFormatter={(v) => `$${Number(v).toFixed(0)}`}
              width={48}
            />
          )}
          {hasProba && (
            <YAxis
              yAxisId="proba"
              orientation="right"
              domain={[probaMin, probaMax]}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
              width={36}
            />
          )}
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              fontSize: 12,
            }}
            formatter={(val: unknown, name: unknown): [string, string] => {
              const n = typeof val === "number" ? val : 0;
              return String(name) === "close"
                ? [`$${n.toFixed(2)}`, "Close"]
                : [`${(n * 100).toFixed(1)}%`, "P(UP)"];
            }}
          />
          {selectedDate && chartData.some((d) => d.date === selectedDate) && (
            <ReferenceLine
              x={selectedDate}
              yAxisId={hasPrices ? "price" : "proba"}
              stroke="hsl(var(--foreground))"
              strokeDasharray="4 4"
              strokeOpacity={0.4}
            />
          )}
          {hasPrices && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="hsl(var(--foreground))"
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
          )}
          {hasProba && (
            <>
              <ReferenceLine
                y={0.5}
                yAxisId="proba"
                stroke="hsl(var(--muted-foreground))"
                strokeDasharray="3 3"
                strokeOpacity={0.35}
              />
              <Area
                yAxisId="proba"
                type="monotone"
                dataKey="proba"
                stroke={chartCfg.color}
                fill={chartCfg.color}
                fillOpacity={0.1}
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-muted-foreground text-center mt-1">{CHART_CLICK_HINT}</p>
      {!hasPrices && hasProba && (
        <p className="text-xs text-muted-foreground text-center mt-1">
          Price line unavailable — showing P(UP) only.
        </p>
      )}
    </div>
  );
}
