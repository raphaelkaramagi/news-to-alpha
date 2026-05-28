import type { EnsembleExplanation, RationaleResponse } from "./types";
import { confidenceFromProba, confidenceLabel } from "./confidence";

function leanDisplay(proba: number): { display: string; direction: "UP" | "DOWN" } {
  const pct = proba * 100;
  if (proba >= 0.5) {
    return { display: `${pct.toFixed(0)}% UP`, direction: "UP" };
  }
  return { display: `${pct.toFixed(0)}% UP`, direction: "DOWN" };
}

const DRIVER_LABELS: Record<string, string> = {
  financial_pred_proba: "Price model",
  lstm_confidence: "Price conviction",
  news_tfidf_pred_proba: "Keywords",
  news_embeddings_pred_proba: "FinBERT",
  spy_return_5d: "Market (5d)",
  all_agree: "Models agree",
};

/** Build explanation in the browser when Flask hasn't been restarted yet. */
export function buildClientExplanation(data: RationaleResponse): EnsembleExplanation | null {
  const proba = data.ensemble_proba ?? data.proba;
  if (proba === null || proba === undefined || Number.isNaN(proba)) {
    return null;
  }

  const hasNews = data.has_news ?? false;
  const route = data.ensemble_route ?? (hasNews ? "has_news" : "no_news");
  const pm = data.per_model ?? {};
  const lstm = pm.lstm?.proba ?? pm.financial?.proba ?? 0.5;
  const tfidf = pm.tfidf?.proba ?? 0.5;
  const emb = pm.embeddings?.proba ?? 0.5;
  const simpleAvg = hasNews ? (lstm + tfidf + emb) / 3 : lstm;
  const direction = proba >= 0.5 ? "UP" : "DOWN";
  const confidence = data.ensemble_confidence ?? confidenceFromProba(proba);
  const label = confidenceLabel(confidence);

  const base_votes: EnsembleExplanation["base_votes"] = [
    { model: "lstm", label: "Price", proba: lstm, active: true, ...leanDisplay(lstm) },
  ];
  if (hasNews) {
    base_votes.push(
      { model: "tfidf", label: "Keywords", proba: tfidf, active: true, ...leanDisplay(tfidf) },
      { model: "embeddings", label: "FinBERT", proba: emb, active: true, ...leanDisplay(emb) }
    );
  } else {
    base_votes.push(
      { model: "tfidf", label: "Keywords", proba: tfidf, active: false, display: "No headlines", direction: "N/A" },
      { model: "embeddings", label: "FinBERT", proba: emb, active: false, display: "No headlines", direction: "N/A" }
    );
  }

  const active = base_votes.filter((v) => v.active !== false);
  const upVotes = active.filter((v) => v.direction === "UP").length;
  const downVotes = active.filter((v) => v.direction === "DOWN").length;

  const headline =
    direction === "DOWN"
      ? `Final call is **DOWN** (${(proba * 100).toFixed(0)}% UP) — below the 50% threshold.`
      : `Final call is **UP** (${(proba * 100).toFixed(0)}% UP) — above the 50% threshold.`;

  const bullets = [headline];
  if (!hasNews) {
    bullets.push("No headlines before 4 PM ET — price-only combiner.");
  } else if (upVotes > 0 && downVotes > 0) {
    bullets.push(
      `Models split **${upVotes} up / ${downVotes} down**; simple avg **${(simpleAvg * 100).toFixed(0)}% UP** → ensemble **${(proba * 100).toFixed(0)}% UP**.`
    );
  }

  const drivers = Object.entries(DRIVER_LABELS).map(([feature, driverLabel]) => ({
    feature,
    label: driverLabel,
    value: 0,
    effect: 0,
    direction: "neutral" as const,
  }));

  return {
    ensemble_proba: proba,
    ensemble_direction: direction,
    ensemble_confidence: confidence,
    confidence_label: label,
    confidence_help: `${(proba * 100).toFixed(0)}% UP is ${label.toLowerCase()} conviction (${(confidence * 100).toFixed(0)}%).`,
    simple_average_proba: simpleAvg,
    summary: bullets[0],
    bullets,
    base_votes,
    drivers,
    news_weight_note: null,
    models_disagree: hasNews && upVotes > 0 && downVotes > 0,
    ensemble_route: route,
    has_news: hasNews,
  };
}
