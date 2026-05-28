"use client";
import { useQuery } from "@tanstack/react-query";
import type { RationaleResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { ModelId } from "@/lib/tickers";
import { MODEL_DISPLAY_LABELS } from "@/lib/models";
import { buildClientExplanation } from "@/lib/ensembleExplainClient";
import { confidenceLabel } from "@/lib/confidence";

interface Props {
  ticker: string;
  date: string;
  model: ModelId;
}

type Explanation = NonNullable<RationaleResponse["explanation"]>;
type Vote = Explanation["base_votes"][number];

async function fetchRationale(
  ticker: string,
  date: string,
  model: string
): Promise<RationaleResponse> {
  const res = await fetch(`/api/rationale?ticker=${ticker}&date=${date}&model=${model}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("fetch failed");
  return res.json();
}

function voteLabel(vote: Vote): string {
  const map: Record<string, string> = {
    lstm: MODEL_DISPLAY_LABELS.lstm,
    tfidf: MODEL_DISPLAY_LABELS.tfidf,
    embeddings: MODEL_DISPLAY_LABELS.embeddings,
  };
  return map[vote.model] ?? vote.label;
}

function humanSummary(exp: Explanation): string {
  const votes = exp.base_votes;
  const up = votes.filter((v) => v.direction === "UP").length;
  const ens = exp.ensemble_direction;
  const pct = (exp.ensemble_proba * 100).toFixed(0);

  if (up === 3) {
    return `Price, keywords, and FinBERT all lean up — combined into a ${ens} call (${pct}% chance of rising).`;
  }
  if (up === 0) {
    return `All three inputs lean down — combined into a ${ens} call (${pct}% up).`;
  }

  const parts = votes.map((v) => {
    const lean = v.direction === "UP" ? "up" : "down";
    return `${voteLabel(v)} ${lean} (${(v.proba * 100).toFixed(0)}%)`;
  });

  const disagree = exp.models_disagree;
  if (disagree) {
    return `Mixed signals: ${parts.join(", ")}. The combiner weighs these and lands on ${ens} (${pct}% up).`;
  }
  return `${parts.join(" · ")} → ${ens} (${pct}% up).`;
}

function VotePill({ vote }: { vote: Vote }) {
  const isUp = vote.direction === "UP";
  return (
    <div className="rounded-md border px-3 py-2 text-center min-w-0 flex-1">
      <p className="text-[11px] text-muted-foreground">{voteLabel(vote)}</p>
      <p className={cn("text-sm font-semibold tabular-nums mt-0.5", isUp ? "text-up" : "text-down")}>
        {(vote.proba * 100).toFixed(0)}% up
      </p>
    </div>
  );
}

export function RationaleBars({ ticker, date, model }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["rationale", ticker, date, model],
    queryFn: () => fetchRationale(ticker, date, model),
    staleTime: 300_000,
    enabled: model === "ensemble",
  });

  if (model !== "ensemble") {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Switch to <span className="font-medium text-foreground">Ensemble</span> to see how the final call was built.
      </p>
    );
  }

  if (isLoading) {
    return <div className="h-28 rounded-lg bg-muted animate-pulse" />;
  }

  if (error || !data) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Explanation isn&apos;t available for this date yet.
      </p>
    );
  }

  const exp = data.explanation ?? buildClientExplanation(data);
  if (!exp) {
    return (
      <p className="text-sm text-muted-foreground py-4">
        Explanation isn&apos;t available for this date yet.
      </p>
    );
  }

  const isUp = exp.ensemble_direction === "UP";
  const confPct = (exp.ensemble_confidence * 100).toFixed(0);
  const label = exp.confidence_label ?? confidenceLabel(exp.ensemble_confidence);
  const hasNews = data.has_news;

  return (
    <div className="space-y-5">
      <div
        className={cn(
          "rounded-lg border px-4 py-3 space-y-2",
          isUp ? "border-up/30 bg-up/5" : "border-down/30 bg-down/5"
        )}
      >
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <p className={cn("text-lg font-semibold", isUp ? "text-up" : "text-down")}>
            {exp.ensemble_direction}
          </p>
          <p className="text-sm text-muted-foreground tabular-nums">
            {(exp.ensemble_proba * 100).toFixed(0)}% chance of going up
          </p>
        </div>
        <p className="text-sm text-muted-foreground">
          {label} conviction ({confPct}%) — how far from a 50/50 split, not odds of being right.
        </p>
      </div>

      <p className="text-sm leading-relaxed">{humanSummary(exp)}</p>

      {hasNews === false && (
        <p className="text-xs text-muted-foreground">
          No headlines this session — the price-only combiner was used.
        </p>
      )}

      <div>
        <p className="text-xs text-muted-foreground mb-2">Input breakdown</p>
        <div className="flex gap-2">
          {exp.base_votes.map((v) => (
            <VotePill key={v.model} vote={v} />
          ))}
        </div>
      </div>
    </div>
  );
}
