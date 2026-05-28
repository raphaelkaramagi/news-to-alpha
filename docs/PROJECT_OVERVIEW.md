# Project overview

Architecture reference for contributors. Operational commands live in [DATA.md](DATA.md). Evaluation metrics in [RESULTS.md](RESULTS.md).

---

## System summary

Predicts **next NYSE session direction** (UP/DOWN) for 20 US tickers by combining:

| Model | Input | Output |
|-------|--------|--------|
| **LSTM** | 60-day price window + ~20 technicals + ticker embedding + SPY/VIX regime | P(UP) |
| **TF-IDF** | Cutoff-aligned headline bigrams (4 PM ET rule) | P(UP) |
| **FinBERT embeddings** | Mean-pooled `ProsusAI/finbert` headline vectors | P(UP) |
| **Ensemble** | HistGradientBoosting on base-model outputs + context features | Final P(UP) |

Optional **conditional ensemble** (`build_ensemble.py --conditional`): separate meta-models for days with vs without headlines.

**Shipped stack:** Next.js UI (`web/`) + Flask JSON API (`app/server.py`). Training is CLI-only — no web-triggered retrain in production (`INFERENCE_ONLY=true`).

---

## Repository layout

```
app/                 Flask API + optional job queue
scripts/             CLI pipeline entry points
src/                 ML library (collection → features → models)
web/                 Next.js dashboard
tests/unit/          pytest suite
docs/                Documentation
data/                Artifacts (gitignored)
```

Complete file-by-file inventory: [PERSONAL_FULL_GUIDE.md](PERSONAL_FULL_GUIDE.md) (operator doc).

---

## Data flow

```
yfinance / Finnhub
       ↓
  SQLite (prices, news, labels)
       ↓
  Feature engineering (technicals, TF-IDF, FinBERT)
       ↓
  Base models (LSTM, TF-IDF LR, embedding classifier)
       ↓
  Ensemble meta-model → final_ensemble_predictions.csv
       ↓
  Flask API reads CSVs + DB → Vercel UI
```

Daily refresh (`daily_update.py`) runs inference-only steps without retraining. See [DATA.md](DATA.md).

---

## Core modules (`src/`)

### Configuration & utilities

| Module | Role |
|--------|------|
| `config.py` | Paths, tickers, hyperparameters, 4 PM ET cutoff |
| `utils/trading_calendar.py` | NYSE sessions, market open/close, forward session |
| `utils/pipeline_config.py` | Persist last train config to JSON |
| `utils/pipeline_cleanup.py` | Prune stale artifacts after retrain |
| `utils/api_clients.py` | Finnhub REST wrapper |

### Database & collection

| Module | Role |
|--------|------|
| `database/schema.py` | SQLite table definitions |
| `data_collection/price_collector.py` | yfinance OHLCV with retry/backoff |
| `data_collection/news_collector.py` | Finnhub headlines, relevance filter, rate limit |
| `data_collection/base_collector.py` | Shared collector base class |

### Processing

| Module | Role |
|--------|------|
| `data_processing/label_generator.py` | Binary labels from next-session close |
| `data_processing/dataset_split.py` | Chronological 70/15/15 split |
| `data_processing/standardization.py` | Date parsing, 4 PM cutoff assignment |
| `data_processing/price_validation.py` | Price data quality checks |
| `data_processing/news_validation.py` | News data quality checks |

### Features

| Module | Role |
|--------|------|
| `features/technical_indicators.py` | RSI, MACD, Bollinger, volume, returns, etc. |
| `features/sequence_generator.py` | 60-day LSTM windows; `generate_live()` for inference |
| `features/lstm_snapshot.py` | LSTM context features for ensemble / UI |
| `features/news_sentiment.py` | FinBERT sentiment pipeline helpers |
| `features/publisher_features.py` | Publisher/source derived features |

### Models

| Module | Role |
|--------|------|
| `models/lstm_model.py` | `StockLSTM` + `LSTMTrainer` (AUC early-stop, calibration) |
| `models/news_pipeline.py` | Shared news dataset builder for TF-IDF and embeddings |

### Inference & explanation

| Module | Role |
|--------|------|
| `ml/lstm_live_export.py` | Append LSTM rows for unlabeled dates |
| `ml/news_live_export.py` | Append TF-IDF + FinBERT rows for live headline days |
| `ml/ensemble_explain.py` | Counterfactual driver bars for Why tab |

### Evaluation

| Module | Role |
|--------|------|
| `evaluation/` | Package stub; metrics logic lives in `scripts/evaluate_predictions.py` |

---

## CLI scripts (`scripts/`)

| Script | Role |
|--------|------|
| `run_pipeline.py` | Full train orchestrator (`--preset`) |
| `daily_update.py` | Collect + infer + ensemble (no retrain) |
| `score_models.py` | Inference-only scoring + outcome backfill |
| `build_ensemble.py` | Train meta-model, write `final_ensemble_predictions.csv` |
| `build_eval_dataset.py` | Join base-model CSVs |
| `evaluate_predictions.py` | Test metrics → [RESULTS.md](RESULTS.md) |
| `publish_deploy_bundle.py` | Trim + SSH upload to Railway `/data` |
| `audit_data_coverage.py` | Price/news/label/live-row audit |
| `collect_prices.py` / `collect_news.py` | Data ingestion |
| `generate_labels.py` / `split_dataset.py` | Labels and split |
| `train_lstm.py` / `train_nlp.py` / `train_news_embeddings.py` | Base model training |
| `setup_database.py` / `reset_data.py` / `validate_data.py` | DB lifecycle |

---

## Web application (`web/`)

Next.js App Router dashboard. Server routes under `app/api/*` proxy to Flask via `API_BASE_URL`.

| Area | Key files |
|------|-----------|
| Pages | `app/page.tsx`, `app/t/[symbol]/`, `app/status/` |
| Markets UI | `components/markets/*` |
| Ticker detail | `components/ticker/*` |
| Charts | `components/charts/*` |
| Shared state | `components/layout/SelectedDateProvider.tsx` |
| API client | `lib/backend.ts`, `lib/types.ts`, `lib/models.ts` |

See [web/README.md](../web/README.md).

---

## API (`app/`)

| File | Role |
|------|------|
| `server.py` | Flask routes: ticker, history, headlines, rationale, metrics, data-status |
| `jobs.py` | Background job queue (disabled when `INFERENCE_ONLY=true`) |

---

## Design decisions

**SQLite** — zero-config persistence; adequate volume for this project.

**Binary classification** — next-session direction is easier to evaluate than point price forecasts. Beating 55% on a chronological test split indicates real signal.

**4 PM ET cutoff** — news after close assigns to the next session; prevents leakage.

**Chronological splits** — no random shuffle; train on past, test on future.

**Conditional ensemble** — separate combiners for headline vs no-headline days improved test accuracy (see [RESULTS.md](RESULTS.md)).

**Confidence formula** — `|P(UP) − 0.5| × 2` measures lean strength, not calibrated correctness.

**Operator-side training** — keeps cloud costs low; API serves precomputed artifacts only.

---

## Known limitations

| Area | Status |
|------|--------|
| LSTM price head | Near-random on test; collapse under investigation |
| News history depth | Finnhub free tier limits backfill |
| Cloud daily infer | Documented but not enabled by default |
| Walk-forward eval | Single chronological split only |

---

## Testing

```bash
pytest tests/unit -q
```

134+ unit tests covering schema, collectors, features, LSTM, news pipeline, validation, ensemble, and evaluation scripts.

---

## Further reading

- [DATA.md](DATA.md) — pipeline operations  
- [RESULTS.md](RESULTS.md) — accuracy and findings  
- [README.md](README.md) — quick start
