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
