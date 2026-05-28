"use client";
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
import { useSelectedDate } from "@/components/layout/SelectedDateProvider";
import { CHART_CLICK_HINT, dateFromChartClick } from "@/lib/chartClick";

export type IndexPoint = { date: string; index: number };

interface Props {
  data: IndexPoint[];
  selectedDate?: string;
}

export function PriceIndexChart({ data, selectedDate }: Props) {
  const { setSelectedDate } = useSelectedDate();

  if (data.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-6 text-center">
        No price data for this window.
      </p>
    );
  }

  const values = data.map((d) => d.index);
  const min = Math.min(...values) * 0.998;
  const max = Math.max(...values) * 1.002;

  return (
    <div className="w-full h-60 min-h-[240px]">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart
          data={data}
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
          <YAxis
            domain={[min, max]}
            tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
            tickFormatter={(v) => Number(v).toFixed(0)}
            width={40}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "6px",
              fontSize: 12,
            }}
            formatter={(v: unknown) => [Number(v).toFixed(1), "Index"]}
          />
          {selectedDate && data.some((d) => d.date === selectedDate) && (
            <ReferenceLine
              x={selectedDate}
              stroke="hsl(var(--foreground))"
              strokeDasharray="4 4"
              strokeOpacity={0.4}
            />
          )}
          <Line
            type="monotone"
            dataKey="index"
            stroke="hsl(var(--foreground))"
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-[10px] text-muted-foreground text-center mt-1">{CHART_CLICK_HINT}</p>
    </div>
  );
}
