import type { EnsembleExplanation, RationaleResponse } from "./types";
import { confidenceFromProba, confidenceLabel } from "./confidence";

function leanDisplay(proba: number): { display: string; direction: "UP" | "DOWN" } {
  const pct = proba * 100;
  if (proba >= 0.5) {
    return { display: `${pct.toFixed(0)}% UP`, direction: "UP" };
  }
  return { display: `${pct.toFixed(0)}% UP`, direction: "DOWN" };
}

// Directional signals (must mirror _DIRECTIONAL_SPECS in ensemble_explain.py).
// The browser fallback has no permutation importances or SPY 5d return, so it
// weights the available base-model leans equally — enough to keep the chart
// sign-correct until Flask serves the full explanation.
const DRIVER_SPECS: { feature: string; label: string; key: "lstm" | "tfidf" | "embeddings" }[] = [
  { feature: "financial_pred_proba", label: "Price model", key: "lstm" },
  { feature: "news_tfidf_pred_proba", label: "Keywords", key: "tfidf" },
  { feature: "news_embeddings_pred_proba", label: "Sentiment", key: "embeddings" },
];

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
  const direction: "UP" | "DOWN" = proba >= 0.5 ? "UP" : "DOWN";
  const confidence = data.ensemble_confidence ?? confidenceFromProba(proba);
  const label = confidenceLabel(confidence);

  const base_votes: EnsembleExplanation["base_votes"] = [
    { model: "lstm", label: "Price", proba: lstm, active: true, ...leanDisplay(lstm) },
  ];
  if (hasNews) {
    base_votes.push(
      { model: "tfidf", label: "Keywords", proba: tfidf, active: true, ...leanDisplay(tfidf) },
      { model: "embeddings", label: "Sentiment", proba: emb, active: true, ...leanDisplay(emb) }
    );
  } else {
    base_votes.push(
      { model: "tfidf", label: "Keywords", proba: tfidf, active: false, display: "No headlines", direction: "N/A" },
      { model: "embeddings", label: "Sentiment", proba: emb, active: false, display: "No headlines", direction: "N/A" }
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

  const driverProbas: Record<"lstm" | "tfidf" | "embeddings", number> = {
    lstm,
    tfidf,
    embeddings: emb,
  };
  // Degraded fallback only (server is the source of truth for Shapley effects):
  // approximate each model's pull as its lean from 50%, in rough probability
  // points. No baseline available client-side, so the UI omits the waterfall.
  const drivers = DRIVER_SPECS.filter((s) => hasNews || s.key === "lstm")
    .map((s) => {
      const p = driverProbas[s.key];
      const effect = (p - 0.5) * 0.3; // ~±15 pts at the extremes
      const direction: "up" | "down" | "neutral" =
        effect > 0.01 ? "up" : effect < -0.01 ? "down" : "neutral";
      return { feature: s.feature, label: s.label, value: p, effect, direction };
    })
    .sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));

  const baseLeanProba = simpleAvg;
  const baseLeanDir: "UP" | "DOWN" = baseLeanProba >= 0.5 ? "UP" : "DOWN";
  const flips = baseLeanDir !== direction;
  const disagreement = {
    base_lean_proba: baseLeanProba,
    base_lean_direction: baseLeanDir,
    ensemble_direction: direction,
    ensemble_proba: proba,
    flips_base_lean: flips,
    flip_drivers: [],
    explanation: flips
      ? `Inputs lean ${baseLeanDir} (${(baseLeanProba * 100).toFixed(0)}% UP) but the ensemble calls ${direction} (${(proba * 100).toFixed(0)}% UP).`
      : `Ensemble call (${direction}) agrees with the input lean.`,
  };

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
    disagreement,
    news_weight_note: null,
    models_disagree: hasNews && upVotes > 0 && downVotes > 0,
    ensemble_route: route,
    has_news: hasNews,
  };
}
