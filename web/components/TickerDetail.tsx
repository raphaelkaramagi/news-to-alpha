"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { ModelId } from "@/lib/tickers";
import { ALLOWED_MODELS } from "@/lib/tickers";
import type { TickerApiResponse } from "@/lib/types";
import { hasMixedSignals } from "@/lib/signals";
import { MixedSignalsBanner } from "@/components/MixedSignalsBanner";
import { PerModelBars } from "@/components/PerModelBars";

const MODEL_LABELS: Record<string, string> = {
  ensemble: "Ensemble",
  lstm: "LSTM",
  tfidf: "TF-IDF",
  embeddings: "Embeddings",
};

type Props = {
  symbol: string;
  initial: TickerApiResponse | null;
  initialError?: string | null;
};

export function TickerDetail({ symbol, initial, initialError }: Props) {
  const [model, setModel] = useState<ModelId>("ensemble");
  const [data, setData] = useState<TickerApiResponse | null>(initial);
  const [error, setError] = useState<string | null>(initialError ?? null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(
    async (m: ModelId) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `/api/ticker?ticker=${encodeURIComponent(symbol)}&model=${encodeURIComponent(m)}`,
          { cache: "no-store" }
        );
        const json = await res.json().catch(() => ({}));
        if (!res.ok) {
          setData(null);
          setError(
            typeof json?.error === "string" ? json.error : `Error ${res.status}`
          );
          return;
        }
        setData(json as TickerApiResponse);
      } catch {
        setData(null);
        setError("Network error");
      } finally {
        setLoading(false);
      }
    },
    [symbol]
  );

  const skipInitialClientFetch = useRef(!!initial);

  useEffect(() => {
    if (initial && model === "ensemble" && skipInitialClientFetch.current) {
      skipInitialClientFetch.current = false;
      return;
    }
    skipInitialClientFetch.current = false;
    void load(model);
  }, [model, load, initial]);

  const mixed = data ? hasMixedSignals(data) : false;
  const confPct =
    data != null ? Math.round((data.confidence ?? 0) * 1000) / 10 : 0;

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="mb-6 flex flex-wrap items-center gap-4">
        <Link
          href="/"
          className="text-sm font-medium text-muted hover:text-accent"
        >
          ← Markets
        </Link>
        <label className="ml-auto flex items-center gap-2 text-sm text-muted">
          <span className="font-medium">Model</span>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as ModelId)}
            className="rounded-lg border border-border bg-surface px-3 py-2 text-foreground outline-none focus:border-accent"
          >
            {ALLOWED_MODELS.map((m) => (
              <option key={m} value={m}>
                {MODEL_LABELS[m] ?? m}
              </option>
            ))}
          </select>
        </label>
      </div>

      {loading && (
        <p className="mb-4 text-sm text-muted" aria-live="polite">
          Loading…
        </p>
      )}

      {error && !data && (
        <div className="rounded-xl border border-down/40 bg-down/10 p-6 text-sm text-down">
          {error}
        </div>
      )}

      {data && (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-4">
            {mixed && <MixedSignalsBanner />}
            <div className="rounded-2xl border border-border bg-surface p-6 shadow-lg">
              <div className="mb-4 flex flex-wrap items-baseline gap-2">
                <h1 className="text-2xl font-bold tracking-tight">{data.ticker}</h1>
                <span className="text-muted">{data.company}</span>
                <span className="rounded-md bg-surface-2 px-2 py-0.5 text-xs font-medium text-muted">
                  {MODEL_LABELS[data.model] ?? data.model}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-6">
                <div
                  className={`text-4xl font-extrabold tracking-tight sm:text-5xl ${
                    data.binary === 1 ? "text-up" : "text-down"
                  }`}
                >
                  {data.binary === 1 ? "▲ Up" : "▼ Down"}
                </div>
                <div className="text-sm leading-relaxed text-muted">
                  <p>
                    <span className="font-semibold text-foreground">
                      {(data.proba * 100).toFixed(1)}%
                    </span>{" "}
                    implied up ·{" "}
                    <span className="text-foreground">Date {data.prediction_date}</span>
                  </p>
                  {data.actual_binary != null && (
                    <p className="mt-2">
                      Outcome:{" "}
                      <span
                        className={
                          data.actual_binary === 1 ? "text-up" : "text-down"
                        }
                      >
                        {data.actual_binary === 1 ? "Up" : "Down"}
                      </span>
                      {data.realized_return != null && (
                        <> · {data.realized_return.toFixed(2)}% move</>
                      )}
                    </p>
                  )}
                </div>
              </div>
              <div className="mt-6">
                <div className="mb-1 flex justify-between text-xs text-muted">
                  <span>Confidence</span>
                  <span className="font-semibold text-foreground">{confPct}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                  <div
                    className={`h-full rounded-full ${
                      confPct >= 60
                        ? "bg-up"
                        : confPct >= 30
                          ? "bg-warn"
                          : "bg-down"
                    }`}
                    style={{ width: `${Math.min(100, confPct)}%` }}
                  />
                </div>
              </div>
            </div>

            {data.top_headlines.length > 0 && (
              <div className="rounded-2xl border border-border bg-surface p-6">
                <h2 className="mb-3 text-sm font-semibold text-muted">
                  Headlines behind this call
                </h2>
                <ul className="space-y-3 text-sm">
                  {data.top_headlines.map((h, i) => (
                    <li
                      key={i}
                      className="border-b border-border/60 pb-3 last:border-0 last:pb-0"
                    >
                      {h}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <div className="rounded-2xl border border-border bg-surface p-6">
              <PerModelBars data={data} />
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-border bg-surface-2/50 p-4 text-xs text-muted">
                <p className="font-semibold text-foreground">Ensemble</p>
                <p className="mt-1">
                  Blends price and news models into a single next-session call.
                </p>
              </div>
              <div className="rounded-xl border border-border bg-surface-2/50 p-4 text-xs text-muted">
                <p className="font-semibold text-foreground">Components</p>
                <p className="mt-1">
                  LSTM on sequences; TF-IDF and embeddings on headlines — compare
                  bars when signals diverge.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
