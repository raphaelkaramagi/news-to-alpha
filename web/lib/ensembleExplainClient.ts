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
  financial_pred_proba: "Price model lean",
  lstm_confidence: "Price conviction",
  news_tfidf_pred_proba: "Keyword headlines",
  news_embeddings_pred_proba: "Headline meaning",
  spy_return_5d: "Broad market (5d)",
  all_agree: "Models agree",
};

/** Build explanation in the browser when Flask hasn't been restarted yet. */
export function buildClientExplanation(data: RationaleResponse): EnsembleExplanation | null {
  const proba = data.ensemble_proba ?? data.proba;
  if (proba === null || proba === undefined || Number.isNaN(proba)) {
    return null;
  }

  const pm = data.per_model ?? {};
  const lstm = pm.lstm?.proba ?? pm.financial?.proba ?? 0.5;
  const tfidf = pm.tfidf?.proba ?? 0.5;
  const emb = pm.embeddings?.proba ?? 0.5;
  const simpleAvg = (lstm + tfidf + emb) / 3;
  const direction = proba >= 0.5 ? "UP" : "DOWN";
  const confidence = data.ensemble_confidence ?? confidenceFromProba(proba);
  const label = confidenceLabel(confidence);

  const base_votes = [
    { model: "lstm", label: "Price", proba: lstm, ...leanDisplay(lstm) },
    { model: "tfidf", label: "Keywords", proba: tfidf, ...leanDisplay(tfidf) },
    { model: "embeddings", label: "Meaning", proba: emb, ...leanDisplay(emb) },
  ];
  const upVotes = base_votes.filter((v) => v.direction === "UP").length;
  const downVotes = 3 - upVotes;

  const headline =
    direction === "DOWN"
      ? `Final call is **DOWN** (${(proba * 100).toFixed(0)}% UP) — below the 50% threshold.`
      : `Final call is **UP** (${(proba * 100).toFixed(0)}% UP) — above the 50% threshold.`;

  const bullets = [headline];
  if (upVotes > 0 && downVotes > 0) {
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
    models_disagree: upVotes > 0 && downVotes > 0,
  };
}
