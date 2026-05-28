import type { ModelId } from "./tickers";

/** Short labels for tabs, vote pills, and tables. */
export const MODEL_DISPLAY_LABELS: Record<ModelId, string> = {
  ensemble: "Ensemble",
  lstm: "Price",
  tfidf: "Keywords",
  embeddings: "FinBERT",
};

export const MODEL_DESCRIPTIONS: Record<
  ModelId,
  { title: string; body: string }
> = {
  ensemble: {
    title: "Ensemble",
    body: "Blends price, keyword, and FinBERT headline signals. When headlines exist, a news-tuned combiner is used.",
  },
  lstm: {
    title: "Price (LSTM)",
    body: "60 days of price action, technicals, and VIX. Ignores headlines.",
  },
  tfidf: {
    title: "Keywords (TF-IDF)",
    body: "Headline word patterns — good when wording is clearly bullish or bearish.",
  },
  embeddings: {
    title: "FinBERT",
    body: "Financial headline meaning (ProsusAI/finbert), averaged per day.",
  },
};
