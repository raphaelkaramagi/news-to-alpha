"use client";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { chartWindowDays, type ChartWindow } from "@/lib/chartWindow";
import type { ModelId } from "@/lib/tickers";
import { MODEL_CHART_CONFIG, MODEL_DISPLAY_LABELS } from "@/lib/models";
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { CHART_CLICK_HINT, dateFromChartClick } from "@/lib/chartClick";

type TracePoint = { date: string; accuracy: number | null };

interface Props {
  ticker?: string;
  window: ChartWindow;
  selectedDate?: string;
  model?: ModelId;
  /** Pre-loaded series (markets overview). Skips fetch when set. */
  series?: TracePoint[];
  isLoading?: boolean;
  label?: string;
}

async function fetchAccuracyTrace(
  ticker: string | undefined,
  model: ModelId,
  rollingWindow: number,
  displayDays: number
): Promise<TracePoint[]> {
  const params = new URLSearchParams({
    window: String(rollingWindow),
    model,
  });
  if (ticker) params.set("ticker", ticker);
  const res = await fetch(`/api/accuracy-trace?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error("fetch failed");
  const data = await res.json();
  const series: TracePoint[] = data.series ?? [];
  return series.filter((p) => p.accuracy !== null).slice(-displayDays);
}

export function AccuracyTraceChart({
  ticker,
  window,
  selectedDate,
  model = "ensemble",
  series: externalSeries,
  isLoading: externalLoading,
  label,
}: Props) {
  const displayDays = chartWindowDays(window);
  const rollingWindow = Math.min(7, Math.max(3, Math.floor(displayDays / 4)));
  const { setSelectedDate } = useSelectedDate();
  const chartCfg = MODEL_CHART_CONFIG[model];
  const traceLabel = label ?? `${MODEL_DISPLAY_LABELS[model]} accuracy`;

  const { data: fetched = [], isLoading: fetchLoading, isError } = useQuery({
    queryKey: ["accuracy-trace", ticker ?? "ALL", model, rollingWindow, displayDays],
    queryFn: () => fetchAccuracyTrace(ticker, model, rollingWindow, displayDays),
    staleTime: 60_000,
    enabled: externalSeries === undefined,
  });

  const chartData = externalSeries ?? fetched;
  const isLoading = externalLoading ?? fetchLoading;

  if (isLoading) {
    return <div className="h-40 w-full rounded-lg bg-muted animate-pulse" />;
  }

  if (isError || chartData.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-6 text-center">
        Not enough resolved history for this window.
      </p>
    );
  }

  return (
    <div className="w-full h-40 min-h-[160px]">
      <ResponsiveContainer width="100%" height={160}>
        <LineChart
          data={chartData}
          margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
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
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
            width={36}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              fontSize: 12,
            }}
            formatter={(v: unknown) => [`${(Number(v) * 100).toFixed(1)}%`, traceLabel]}
          />
          <ReferenceLine y={0.5} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" strokeOpacity={0.35} />
          {selectedDate && chartData.some((d) => d.date === selectedDate) && (
            <ReferenceLine
              x={selectedDate}
              stroke="hsl(var(--foreground))"
              strokeDasharray="4 4"
              strokeOpacity={0.4}
            />
          )}
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke={chartCfg.color}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-muted-foreground text-center mt-1">{CHART_CLICK_HINT}</p>
    </div>
  );
}
