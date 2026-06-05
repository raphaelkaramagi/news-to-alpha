import type { EnsembleExplanation } from "./types";
import { MODEL_DISPLAY_LABELS } from "./models";

type Vote = EnsembleExplanation["base_votes"][number];

function voteLabel(vote: Vote): string {
  const map: Record<string, string> = {
    lstm: MODEL_DISPLAY_LABELS.lstm,
    tfidf: MODEL_DISPLAY_LABELS.tfidf,
    embeddings: MODEL_DISPLAY_LABELS.embeddings,
  };
  return map[vote.model] ?? vote.label;
}

export function routeNote(route: string | null | undefined): string | null {
  if (route === "has_news") return "Headlines present — news-tuned combiner.";
  if (route === "no_news") return "No headlines — price-only combiner.";
  return null;
}

/** Plain-English summary for the Why tab (layperson). */
export function humanEnsembleSummary(
  exp: EnsembleExplanation,
  hasNews?: boolean,
  route?: string | null
): string {
  const votes = exp.base_votes;
  const active = votes.filter((v) => v.active !== false && v.direction !== "N/A");
  const ens = exp.ensemble_direction;
  const pct = (exp.ensemble_proba * 100).toFixed(0);

  if (hasNews === false || route === "no_news") {
    const price = votes.find((v) => v.model === "lstm");
    const lean = price?.direction === "UP" ? "up" : "down";
    const pricePct = price ? (price.proba * 100).toFixed(0) : "50";
    return `No headlines before market close, so only the price model ran (${pricePct}% up). The price-only combiner landed on ${ens} (${pct}% chance of rising).`;
  }

  const up = active.filter((v) => v.direction === "UP").length;
  if (up === active.length && active.length > 0) {
    return `Price, keywords, and sentiment all lean up — the news-tuned combiner calls ${ens} (${pct}% chance of rising).`;
  }
  if (up === 0 && active.length > 0) {
    return `All active models lean down — combined into ${ens} (${pct}% up).`;
  }

  const parts = active.map((v) => {
    const lean = v.direction === "UP" ? "up" : "down";
    return `${voteLabel(v)} ${lean} (${(v.proba * 100).toFixed(0)}%)`;
  });

  if (exp.models_disagree && parts.length > 1) {
    return `Mixed signals: ${parts.join(", ")}. The news-tuned combiner weighs these and lands on ${ens} (${pct}% up).`;
  }
  return `${parts.join(" · ")} → ${ens} (${pct}% up).`;
}
