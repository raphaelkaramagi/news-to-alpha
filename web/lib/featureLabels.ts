/** Client-side labels (mirrors app/server.py — used when API omits label/hint). */

export const FEATURE_LABELS: Record<string, string> = {
  financial_pred_proba: "Price trend (LSTM)",
  news_tfidf_pred_proba: "Headline keywords",
  news_embeddings_pred_proba: "Headline meaning",
  lstm_confidence: "Price model confidence",
  tfidf_confidence: "Keyword model confidence",
  emb_confidence: "Embedding model confidence",
  all_agree: "Models agree",
  has_news: "News that day",
  n_headlines: "Headline count",
  spy_return_5d: "Overall market (5 days)",
};

export const FEATURE_HINTS: Record<string, string> = {
  financial_pred_proba: "Reads 60 days of price action and technicals.",
  news_tfidf_pred_proba: "Word patterns in headlines (TF-IDF model).",
  news_embeddings_pred_proba: "Semantic meaning of headlines (embedding model).",
  spy_return_5d: "Recent direction of the broad market (SPY).",
  has_news: "Whether any headlines were collected for this date.",
  n_headlines: "How many headlines fed the news models.",
  all_agree: "Whether LSTM, TF-IDF, and embeddings all agree.",
};

export function featureLabel(name: string): string {
  return FEATURE_LABELS[name] ?? name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function featureHint(name: string): string {
  return FEATURE_HINTS[name] ?? "";
}
