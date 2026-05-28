import type { ModelId } from "./tickers";

/** Short labels for tabs, vote pills, and tables. */
export const MODEL_DISPLAY_LABELS: Record<ModelId, string> = {
  ensemble: "Ensemble",
  lstm: "Price",
  tfidf: "Keywords",
  embeddings: "FinBERT",
};

/** Chart styling + legend copy per model tab. */
export const MODEL_CHART_CONFIG: Record<
  ModelId,
  { color: string; probaLegend: string; accuracyLegend: string }
> = {
  ensemble: {
    color: "hsl(var(--up))",
    probaLegend: "Black = close · green = Ensemble P(UP) (right axis)",
    accuracyLegend: "Ensemble rolling hit rate · dashed = 50%",
  },
  lstm: {
    color: "hsl(217 91% 60%)",
    probaLegend: "Black = close · blue = Price (LSTM) P(UP) (right axis)",
    accuracyLegend: "Price (LSTM) rolling hit rate · dashed = 50%",
  },
  tfidf: {
    color: "hsl(25 95% 53%)",
    probaLegend: "Black = close · orange = Keywords (TF-IDF) P(UP) (right axis)",
    accuracyLegend: "Keywords (TF-IDF) rolling hit rate · dashed = 50%",
  },
  embeddings: {
    color: "hsl(271 81% 56%)",
    probaLegend: "Black = close · purple = FinBERT P(UP) (right axis)",
    accuracyLegend: "FinBERT rolling hit rate · dashed = 50%",
  },
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
