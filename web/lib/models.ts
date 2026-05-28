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
    body: "Combines three inputs: Price (LSTM), Keywords (TF-IDF), and FinBERT. With headlines, a news-tuned combiner runs; without headlines, a price-only combiner is used.",
  },
  lstm: {
    title: "Price (LSTM)",
    body: "Reads 60 days of price action, technical indicators, VIX, and SPY context. Does not use headlines.",
  },
  tfidf: {
    title: "Keywords (TF-IDF)",
    body: "Scores headline wording (bigram patterns). Only runs when at least one headline exists before 4 PM ET.",
  },
  embeddings: {
    title: "FinBERT",
    body: "Scores financial sentiment from headline meaning (ProsusAI/finbert). Only runs when headlines are present.",
  },
};
