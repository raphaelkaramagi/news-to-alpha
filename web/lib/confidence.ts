/** Matches backend: abs(proba - 0.5) * 2 — distance from a 50/50 split. */
export function confidenceFromProba(proba: number): number {
  return Math.abs(proba - 0.5) * 2;
}

export function confidenceLabel(confidence: number): "Low" | "Moderate" | "Strong" {
  if (confidence < 0.25) return "Low";
  if (confidence < 0.45) return "Moderate";
  return "Strong";
}

export const CONFIDENCE_HELP =
  "How far the forecast is from 50/50 — not the chance the call is right. " +
  "27% UP is a stronger lean than 45% UP, so confidence can be high even on a DOWN call.";
