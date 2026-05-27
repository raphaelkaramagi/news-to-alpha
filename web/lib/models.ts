import type { ModelId } from "./tickers";

export const MODEL_DESCRIPTIONS: Record<
  ModelId,
  { title: string; body: string }
> = {
  ensemble: {
    title: "Ensemble",
    body: "Combines price, keyword, and embedding models with a learned meta-model. Use this for the final call and the “Why this call” breakdown.",
  },
  lstm: {
    title: "LSTM (price)",
    body: "Looks at 60 trading days of price and technical indicators (RSI, MACD, volume, etc.). Ignores headlines — useful when news is thin or noisy.",
  },
  tfidf: {
    title: "TF-IDF (headlines)",
    body: "Treats headline word patterns as features (bigrams + logistic regression). Fast baseline; strong when wording is clearly bullish or bearish.",
  },
  embeddings: {
    title: "Embeddings (headlines)",
    body: "Encodes headline meaning with MiniLM sentence embeddings, averaged per day. Captures semantics beyond exact keywords.",
  },
};
