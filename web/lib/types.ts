import type { ModelId } from "./tickers";

export type PerModelEntry = {
  proba: number;
  binary: number;
  confidence: number;
};

/** Close(T) → close(T+1) price context for a forecast session. */
export type PriceContext = {
  session_date: string;
  target_date: string | null;
  /** Scoring uses these two closes only (not same-day open→close). */
  validation_basis: "close_to_close";
  start_close_date: string;
  start_close: number | null;
  end_close_date: string | null;
  end_close: number | null;
  session_open: number | null;
  session_close: number | null;
  target_open: number | null;
  target_close: number | null;
  return_pct: number | null;
  resolved: boolean;
  actual_direction: "up" | "down" | null;
  horizon_label: string;
};

/** Response from Flask `GET /api/ticker` */
export type TickerApiResponse = {
  ticker: string;
  company: string;
  prediction_date: string;
  forecast_date?: string | null;
  model: ModelId | string;
  proba: number;
  binary: number;
  confidence: number;
  actual_binary: number | null;
  hit: number | null;
  realized_return: number | null;
  top_headlines: string[];
  per_model: Record<string, PerModelEntry>;
  price_context?: PriceContext;
  /** Predicted next-session |return| in percent (volatility model). */
  expected_move_pct?: number | null;
  actual_abs_return_pct?: number | null;
  forecast_low?: number | null;
  forecast_high?: number | null;
};

export type DataStatus = {
  today: string;
  latest_prediction_date: string | null;
  /** Default date for Markets / date picker — latest price session, not forward preview. */
  primary_prediction_date?: string | null;
  latest_price_date: string | null;
  latest_news_date: string | null;
  latest_resolved_prediction_date: string | null;
  prediction_rows: number;
  price_rows: number;
  news_rows: number;
  last_trading_session: string | null;
  expected_latest_prediction_date: string | null;
  trading_sessions_behind: number;
  is_current: boolean;
  market_status: "open" | "closed" | "pre_market" | null;
  pending_reason: "awaiting_next_close" | "awaiting_data_refresh" | "resolved" | null;
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
  disagreement?: {
    base_lean_proba: number;
    base_lean_direction: "UP" | "DOWN";
    ensemble_direction: "UP" | "DOWN";
    ensemble_proba: number;
    flips_base_lean: boolean;
    flip_drivers: Array<{
      feature: string;
      label: string;
      value: number;
      effect: number;
      direction: "up" | "down" | "neutral";
    }>;
    explanation: string;
  };
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
  open?: number | null;
  high?: number | null;
  low?: number | null;
  close: number;
  volume?: number | null;
};

export type HistoryResponse = {
  ticker: string;
  window: number | "all";
  prices: PricePoint[];
  predictions: HistoryPoint[];
};

export type AccuracySummary = {
  ticker?: string;
  scope?: string;
  window: string;
  n: number;
  hits: number;
  accuracy: number | null;
  rows?: Array<{
    date: string;
    ticker?: string;
    pred_binary?: number | null;
    actual_binary?: number | null;
    hit: number;
    return?: number | null;
    proba?: number | null;
  }>;
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
  session_close?: number | null;
  target_close?: number | null;
  target_date?: string | null;
};
