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
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { CHART_CLICK_HINT, dateFromChartClick } from "@/lib/chartClick";
import { fetchVolatilityTrace } from "@/lib/volatilityAccuracy";

const VOL_CHART_COLOR = "hsl(199 89% 48%)";

type TracePoint = { date: string; accuracy: number | null };

interface Props {
  ticker?: string;
  window: ChartWindow;
  selectedDate?: string;
  series?: TracePoint[];
  isLoading?: boolean;
}

export function VolatilityTraceChart({
  ticker,
  window,
  selectedDate,
  series: externalSeries,
  isLoading: externalLoading,
}: Props) {
  const displayDays = chartWindowDays(window);
  const rollingWindow = Math.min(7, Math.max(3, Math.floor(displayDays / 4)));
  const { setSelectedDate } = useSelectedDate();

  const { data: fetched = [], isLoading: fetchLoading, isError } = useQuery({
    queryKey: ["volatility-trace", ticker ?? "ALL", rollingWindow, displayDays],
    queryFn: () => fetchVolatilityTrace(ticker, rollingWindow, displayDays),
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
            formatter={(v: unknown) => [
              `${(Number(v) * 100).toFixed(1)}%`,
              "Within ± band",
            ]}
          />
          <ReferenceLine
            y={0.5}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="3 3"
            strokeOpacity={0.35}
          />
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
            stroke={VOL_CHART_COLOR}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-muted-foreground text-center mt-1">{CHART_CLICK_HINT}</p>
    </div>
  );
}
