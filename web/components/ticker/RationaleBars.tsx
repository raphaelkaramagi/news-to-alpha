"use client";
import { useQuery } from "@tanstack/react-query";
import type { RationaleResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import type { ModelId } from "@/lib/tickers";
import { MODEL_DISPLAY_LABELS } from "@/lib/models";
import { buildClientExplanation } from "@/lib/ensembleExplainClient";
import { confidenceLabel } from "@/lib/confidence";
import { humanEnsembleSummary, routeNote } from "@/lib/ensembleSummary";

interface Props {
  ticker: string;
  date: string;
  model: ModelId;
}

type Vote = RationaleResponse["explanation"] extends infer E
  ? E extends { base_votes: infer V }
    ? V extends Array<infer U>
      ? U
      : never
    : never
  : never;

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

function VotePill({ vote }: { vote: Vote }) {
  const inactive = vote.active === false || vote.direction === "N/A";
  const isUp = !inactive && vote.direction === "UP";
  return (
    <div
      className={cn(
        "rounded-md border px-3 py-2 text-center min-w-0 flex-1",
        inactive && "opacity-60"
      )}
    >
      <p className="text-[11px] text-muted-foreground">{voteLabel(vote)}</p>
      <p
        className={cn(
          "text-sm font-semibold tabular-nums mt-0.5",
          inactive ? "text-muted-foreground" : isUp ? "text-up" : "text-down"
        )}
      >
        {inactive ? "No headlines" : `${(vote.proba * 100).toFixed(0)}% up`}
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
        Switch to <span className="font-medium text-foreground">Ensemble</span> to see how the final call was built from three inputs: price, keywords, and FinBERT.
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
  const hasNews = data.has_news ?? exp.has_news;
  const route = data.ensemble_route ?? exp.ensemble_route;
  const routeText = routeNote(route);

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

      {routeText && (
        <p className="text-xs text-muted-foreground border-l-2 border-muted pl-3">{routeText}</p>
      )}

      <p className="text-sm leading-relaxed">
        {humanEnsembleSummary(exp, hasNews, route)}
      </p>

      <div>
        <p className="text-xs text-muted-foreground mb-1">The three inputs</p>
        <p className="text-[11px] text-muted-foreground mb-2">
          <span className="font-medium text-foreground">Price</span> — 60 days of charts &amp; indicators.
          {" "}<span className="font-medium text-foreground">Keywords</span> — headline word patterns.
          {" "}<span className="font-medium text-foreground">FinBERT</span> — financial meaning of headlines.
        </p>
        <div className="flex gap-2">
          {exp.base_votes.map((v) => (
            <VotePill key={v.model} vote={v} />
          ))}
        </div>
      </div>
    </div>
  );
}
