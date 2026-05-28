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
} from "recharts";
import type { ModelId } from "@/lib/tickers";
import { MODEL_DISPLAY_LABELS } from "@/lib/models";
import type { PerModelEntry, RationaleResponse, TickerApiResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import { buildClientExplanation } from "@/lib/ensembleExplainClient";
import { routeNote } from "@/lib/ensembleSummary";
import { ExplanationDrivers } from "@/components/ticker/ExplanationDrivers";
import { EnsembleInputsTable, LstmContextTable } from "@/components/ticker/EnsembleInputsTable";

interface Props {
  ticker: string;
  date: string;
  model: ModelId;
  perModel: Record<string, PerModelEntry>;
  tickerData: TickerApiResponse;
}

type TracePoint = { date: string; accuracy: number | null };

async function fetchAccuracyTrace(ticker: string, window = 30): Promise<TracePoint[]> {
  const res = await fetch(
    `/api/accuracy-trace?ticker=${ticker}&window=${window}`,
    { cache: "no-store" }
  );
  if (!res.ok) throw new Error("fetch failed");
  const data = await res.json();
  return data.series ?? [];
}

async function fetchRationale(ticker: string, date: string): Promise<RationaleResponse | null> {
  const res = await fetch(`/api/rationale?ticker=${ticker}&date=${date}&model=ensemble`, {
    cache: "no-store",
  });
  if (!res.ok) return null;
  return res.json();
}

function routeNoteFromData(data: RationaleResponse | null | undefined): string | null {
  return routeNote(data?.ensemble_route ?? data?.explanation?.ensemble_route);
}

export function AdvancedPanel({ ticker, date, model, perModel, tickerData }: Props) {
  const { data: trace = [], isLoading } = useQuery({
    queryKey: ["accuracy-trace", ticker, 30],
    queryFn: () => fetchAccuracyTrace(ticker, 30),
    staleTime: 300_000,
  });

  const { data: rationale } = useQuery({
    queryKey: ["rationale-advanced", ticker, date],
    queryFn: () => fetchRationale(ticker, date),
    staleTime: 300_000,
    enabled: model === "ensemble",
  });

  const explanation =
    rationale?.explanation ??
    (rationale ? buildClientExplanation(rationale) : null);
  const drivers = explanation?.drivers ?? [];
  const newsNote = explanation?.news_weight_note ?? null;

  const chartData = trace.filter((p) => p.accuracy !== null).slice(-60);

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-medium mb-3">Per-model probabilities ({date})</p>
        <div className="rounded-lg border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/40 text-left text-xs text-muted-foreground">
                <th className="px-3 py-2 font-medium">Model</th>
                <th className="px-3 py-2 font-medium">P(UP)</th>
                <th className="px-3 py-2 font-medium">Call</th>
                <th className="px-3 py-2 font-medium">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(perModel).map(([key, entry]) => (
                <tr
                  key={key}
                  className={cn(
                    "border-b last:border-0",
                    key === model && "bg-muted/30"
                  )}
                >
                  <td className="px-3 py-2 font-medium">
                    {MODEL_DISPLAY_LABELS[key as keyof typeof MODEL_DISPLAY_LABELS] ?? key}
                  </td>
                  <td className="px-3 py-2 tabular-nums">{(entry.proba * 100).toFixed(1)}%</td>
                  <td className={cn(
                    "px-3 py-2 font-medium",
                    entry.binary === 1 ? "text-up" : "text-down"
                  )}>
                    {entry.binary === 1 ? "UP" : "DOWN"}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-muted-foreground">
                    {(entry.confidence * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {model === "ensemble" && (
        <>
          <div>
            <p className="text-sm font-medium mb-2">Ensemble inputs (all 13 features)</p>
            <p className="text-xs text-muted-foreground mb-3">
              What the combiner actually sees — model scores, headline flags, SPY regime, and interaction terms.
            </p>
            <EnsembleInputsTable features={rationale?.meta_features ?? []} />
          </div>

          <LstmContextTable context={rationale?.lstm_context ?? null} />

          <div>
            <p className="text-sm font-medium mb-3">What moved the score</p>
            <ExplanationDrivers
              drivers={drivers}
              newsNote={newsNote}
              routeNote={routeNoteFromData(rationale)}
            />
          </div>
        </>
      )}

      <div>
        <p className="text-sm font-medium mb-1">Rolling accuracy (30-day window)</p>
        <p className="text-xs text-muted-foreground mb-3">
          Share of correct ensemble calls over the prior 30 resolved sessions.
        </p>
        {isLoading ? (
          <div className="h-40 rounded-lg bg-muted animate-pulse" />
        ) : chartData.length === 0 ? (
          <p className="text-sm text-muted-foreground py-6 text-center">
            Not enough resolved history yet.
          </p>
        ) : (
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.4} />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v) => String(v).slice(5)}
                  interval="preserveStartEnd"
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                  width={36}
                />
                <Tooltip
                  formatter={(v: unknown) => [
                    `${(Number(v) * 100).toFixed(1)}%`,
                    "Accuracy",
                  ]}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="hsl(var(--foreground))"
                  strokeWidth={1.5}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div className="rounded-lg border px-4 py-3 text-sm space-y-2">
        <p className="font-medium">Call metadata</p>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
          <div>
            <dt className="text-muted-foreground">Ensemble P(UP)</dt>
            <dd className="font-mono tabular-nums">{(tickerData.proba * 100).toFixed(2)}%</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Resolved</dt>
            <dd>{tickerData.actual_binary === null ? "Pending" : tickerData.hit === 1 ? "Correct" : "Wrong"}</dd>
          </div>
          {tickerData.realized_return !== null && (
            <div>
              <dt className="text-muted-foreground">Realized move</dt>
              <dd className={cn(
                "font-mono tabular-nums",
                tickerData.realized_return >= 0 ? "text-up" : "text-down"
              )}>
                {tickerData.realized_return >= 0 ? "+" : ""}
                {tickerData.realized_return.toFixed(2)}%
              </dd>
            </div>
          )}
          <div>
            <dt className="text-muted-foreground">View</dt>
            <dd>{MODEL_DISPLAY_LABELS[model] ?? model}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
