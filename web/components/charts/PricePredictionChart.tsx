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
  Dot,
} from "recharts";
import type { HistoryResponse } from "@/lib/types";
import type { ChartWindow } from "@/lib/chartWindow";
import { chartWindowDays } from "@/lib/chartWindow";
import type { ModelId } from "@/lib/tickers";
import { MODEL_CHART_CONFIG } from "@/lib/models";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { CHART_CLICK_HINT, dateFromChartClick } from "@/lib/chartClick";
import { formatPrice, shortDate } from "@/lib/forecastHorizon";

interface Props {
  ticker: string;
  selectedDate: string;
  targetDate?: string | null;
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
  if (span < 0.15) {
    const pad = Math.max(0.05, span * 0.5);
    return [Math.max(0, min - pad), Math.min(1, max + pad)];
  }
  return [0, 1];
}

type ChartRow = {
  date: string;
  open: number | null;
  close: number | null;
  proba: number | null;
  isSession: boolean;
  isTarget: boolean;
};

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartRow }>;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div
      className="rounded-md border bg-card px-3 py-2 text-xs shadow-sm"
      style={{ fontSize: 12 }}
    >
      <p className="font-medium mb-1">{row.date}</p>
      {row.close != null && (
        <p className="font-mono tabular-nums">
          Close {formatPrice(row.close)}
          {row.open != null && (
            <span className="text-muted-foreground ml-2 text-[10px]">
              (open {formatPrice(row.open)}, not used for scoring)
            </span>
          )}
        </p>
      )}
      {row.proba != null && (
        <p className="text-muted-foreground mt-0.5">
          P(UP) {(row.proba * 100).toFixed(1)}%
        </p>
      )}
      {row.isSession && (
        <p className="text-[10px] text-muted-foreground mt-1">Start close (day T)</p>
      )}
      {row.isTarget && (
        <p className="text-[10px] text-muted-foreground mt-1">End close (day T+1, outcome)</p>
      )}
    </div>
  );
}

export function PricePredictionChart({
  ticker,
  selectedDate,
  targetDate,
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
    if (!data) return [] as ChartRow[];
    const priceMap = new Map(data.prices.map((p) => [p.date, p]));
    const predMap = new Map(data.predictions.map((p) => [p.prediction_date, p]));
    const allDates = Array.from(
      new Set([
        ...data.prices.map((p) => p.date),
        ...data.predictions.map((p) => p.prediction_date),
      ])
    ).sort();
    return allDates
      .map((date) => {
        const price = priceMap.get(date);
        const pred = predMap.get(date);
        const raw = pred
          ? (pred as unknown as Record<string, number | null>)[probaCol]
          : null;
        return {
          date,
          open: price?.open ?? null,
          close: price?.close ?? null,
          proba: raw != null && !Number.isNaN(raw) ? raw : null,
          isSession: date === selectedDate,
          isTarget: !!targetDate && date === targetDate,
        };
      })
      .slice(-days);
  }, [data, days, probaCol, selectedDate, targetDate]);

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

  const sessionMarker = selectedDate && chartData.some((d) => d.date === selectedDate);
  const targetMarker = targetDate && chartData.some((d) => d.date === targetDate);

  return (
    <div className="w-full h-60 min-h-[240px]">
      {(sessionMarker || targetMarker) && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-muted-foreground mb-2 px-1">
          {sessionMarker && (
            <span>
              <span className="inline-block w-3 border-t-2 border-dashed border-foreground/50 align-middle mr-1" />
              {shortDate(selectedDate)} close (start)
            </span>
          )}
          {targetMarker && (
            <span>
              <span className="inline-block w-3 border-t-2 border-dashed border-up/60 align-middle mr-1" />
              {shortDate(targetDate!)} close (end / outcome)
            </span>
          )}
        </div>
      )}
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
          <Tooltip content={<ChartTooltip />} />
          {sessionMarker && (
            <ReferenceLine
              x={selectedDate}
              yAxisId={hasPrices ? "price" : "proba"}
              stroke="hsl(var(--foreground))"
              strokeDasharray="4 4"
              strokeOpacity={0.45}
            />
          )}
          {targetMarker && (
            <ReferenceLine
              x={targetDate!}
              yAxisId={hasPrices ? "price" : "proba"}
              stroke="hsl(var(--chart-up, 142 71% 45%))"
              strokeDasharray="2 3"
              strokeOpacity={0.55}
            />
          )}
          {hasPrices && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="hsl(var(--foreground))"
              strokeWidth={1.5}
              dot={(props) => {
                const { cx, cy, payload } = props as {
                  cx?: number;
                  cy?: number;
                  payload?: ChartRow;
                };
                if (cx == null || cy == null || !payload) return null;
                if (payload.isSession) {
                  return (
                    <Dot
                      cx={cx}
                      cy={cy}
                      r={4}
                      fill="hsl(var(--foreground))"
                      stroke="hsl(var(--background))"
                      strokeWidth={2}
                    />
                  );
                }
                if (payload.isTarget) {
                  return (
                    <Dot
                      cx={cx}
                      cy={cy}
                      r={4}
                      fill="hsl(var(--chart-up, 142 71% 45%))"
                      stroke="hsl(var(--background))"
                      strokeWidth={2}
                    />
                  );
                }
                return null;
              }}
              activeDot={{ r: 4 }}
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
