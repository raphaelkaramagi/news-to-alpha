/** Mirrors src/config.py — keep in sync when the universe changes. */
export const TICKERS: readonly string[] = [
  "AAPL",
  "NVDA",
  "WMT",
  "LLY",
  "JPM",
  "XOM",
  "MCD",
  "TSLA",
  "DAL",
  "MAR",
  "GS",
  "NFLX",
  "META",
  "ORCL",
  "PLTR",
] as const;

export const TICKER_TO_COMPANY: Record<string, string> = {
  AAPL: "Apple",
  NVDA: "NVIDIA",
  WMT: "Walmart",
  LLY: "Eli Lilly",
  JPM: "JPMorgan Chase",
  XOM: "Exxon Mobil",
  MCD: "McDonald's",
  TSLA: "Tesla",
  DAL: "Delta Air Lines",
  MAR: "Marriott International",
  GS: "Goldman Sachs Group",
  NFLX: "Netflix",
  META: "Meta",
  ORCL: "Oracle",
  PLTR: "Palantir",
};

export const ALLOWED_MODELS = [
  "ensemble",
  "lstm",
  "tfidf",
  "embeddings",
] as const;

export type ModelId = (typeof ALLOWED_MODELS)[number];
