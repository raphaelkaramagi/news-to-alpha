import type { ModelId } from "./tickers";

export type PerModelEntry = {
  proba: number;
  binary: number;
  confidence: number;
};

/** Response from Flask `GET /api/ticker` */
export type TickerApiResponse = {
  ticker: string;
  company: string;
  prediction_date: string;
  model: ModelId | string;
  proba: number;
  binary: number;
  confidence: number;
  actual_binary: number | null;
  hit: number | null;
  realized_return: number | null;
  top_headlines: string[];
  per_model: Record<string, PerModelEntry>;
};

export type DataStatus = {
  today: string;
  latest_prediction_date: string | null;
  latest_price_date: string | null;
  latest_news_date: string | null;
  prediction_rows: number;
  price_rows: number;
  news_rows: number;
  last_trading_session: string | null;
  expected_latest_prediction_date: string | null;
  trading_sessions_behind: number;
  is_current: boolean;
  last_published_at: string | null;
  deploy_mode: "inference_only" | "full";
  horizon?: number;
  train_config?: {
    encoder_model?: string;
    conditional_ensemble?: boolean;
    min_move_pct?: number;
    lstm_epochs?: number;
  };
};

export type JobSpec = {
  id: string;
  kind: string;
  label: string;
  status: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: string | null;
  error: string | null;
  log: string[];
};

export type JobsResponse = {
  current: JobSpec | null;
  recent: JobSpec[];
};

export type HeadlineCard = {
  headline?: string;
  title?: string;
  url: string | null;
  published_at: string | null;
  source: string | null;
  finnhub_sentiment?: number | null;
  sentiment_finnhub?: number | null;
  relevance_score?: number | null;
  relevance?: number | null;
  content?: string;
};

export type HeadlinesResponse = {
  ticker: string;
  date: string;
  headlines: HeadlineCard[];
};

export type RationaleFeature = {
  feature: string;
  label?: string;
  hint?: string;
  value: number;
  contribution: number;
  importance?: number;
  direction?: "up" | "down" | "neutral";
};

export type EnsembleExplanation = {
  ensemble_proba: number;
  ensemble_direction: "UP" | "DOWN";
  ensemble_confidence: number;
  confidence_label?: "Low" | "Moderate" | "Strong";
  confidence_help?: string;
  simple_average_proba?: number;
  summary: string;
  bullets?: string[];
  base_votes: Array<{
    model: string;
    label: string;
    proba: number;
    display: string;
    direction: "UP" | "DOWN" | "N/A";
    active?: boolean;
  }>;
  drivers: Array<{
    feature: string;
    label: string;
    value: number;
    effect: number;
    direction: "up" | "down" | "neutral";
  }>;
  news_weight_note?: string | null;
  models_disagree?: boolean;
  ensemble_route?: "has_news" | "no_news" | "unified" | null;
  has_news?: boolean;
};

export type RationaleResponse = {
  ticker: string;
  date: string;
  model?: string;
  proba?: number;
  ensemble_proba?: number | null;
  ensemble_confidence?: number | null;
  per_model?: Record<string, PerModelEntry>;
  features?: RationaleFeature[];
  contributions?: RationaleFeature[];
  explanation?: EnsembleExplanation;
  has_news?: boolean;
  ensemble_route?: "has_news" | "no_news" | "unified" | null;
  temperature?: number;
  meta_features?: MetaFeatureRow[];
  lstm_context?: LstmContextSnapshot | null;
};

export type MetaFeatureRow = {
  key: string;
  label: string;
  hint?: string;
  value: number;
  importance?: number;
};

export type LstmContextField = {
  key: string;
  label: string;
  value: number;
  unit: string;
};

export type LstmContextSnapshot = {
  available: boolean;
  note?: string;
  fields: LstmContextField[];
};

export type HistoryPoint = {
  prediction_date: string;
  ensemble_pred_proba: number | null;
  financial_pred_proba: number | null;
  news_tfidf_pred_proba: number | null;
  news_embeddings_pred_proba: number | null;
  ensemble_pred_binary: number | null;
  actual_binary: number | null;
};

export type PricePoint = {
  date: string;
  close: number;
};

export type HistoryResponse = {
  ticker: string;
  window: number | "all";
  prices: PricePoint[];
  predictions: HistoryPoint[];
};

export type AccuracySummary = {
  ticker: string;
  window: string;
  n: number;
  hits: number;
  accuracy: number | null;
};

export type ConvictionBucket = {
  confidence_min: number;
  confidence_max: number;
  n: number;
  accuracy: number | null;
};

export type MetricsResponse = {
  overall: Array<{
    model: string;
    split: string;
    subset: string;
    accuracy: number;
    auc: number;
    n: number;
  }>;
  by_ticker: unknown[];
};

export type LastResolvedRow = {
  date: string;
  proba: number | null;
  pred_binary: number | null;
  actual_binary: number | null;
  hit: number | null;
  return: number | null;
};
