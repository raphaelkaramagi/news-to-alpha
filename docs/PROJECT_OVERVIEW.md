# Project Overview

## What It Does

Predicts whether a stock goes **up or down** the next trading day using two models:

1. **LSTM** — learns from 60-day price windows + technical indicators (RSI, MACD, etc.)
2. **NLP** — learns from news headlines (TF-IDF baseline → FinBERT later)
3. **Ensemble** — combines both for a final prediction

End goal: a web app where you search a ticker and see a prediction with confidence + the headlines behind it.

---

## How to Test

```bash
source .venv/bin/activate
```

### Easiest: one-command demo

Runs the full pipeline — collect prices, collect news, generate labels, split,
build features, train LSTM, train NLP baseline — all in one go:

```bash
python scripts/demo.py --reset                      # fresh start
python scripts/demo.py                              # re-run (skips duplicates)
python scripts/demo.py --skip-training              # data pipeline only
```

### Customize tickers and data volume

```bash
python scripts/demo.py --tickers AAPL NVDA TSLA     # specific tickers
python scripts/demo.py --all                         # all 15 project tickers
python scripts/demo.py --all --days 365              # all tickers, 1 year
python scripts/demo.py --days 500                    # 2 years of history
```

### Step by step (for more control)

```bash
# Data collection
python scripts/setup_database.py
python scripts/collect_prices.py --tickers AAPL TSLA     # 365 days by default
python scripts/collect_news.py --tickers AAPL --days 30

# Labels + splits
python scripts/generate_labels.py
python scripts/split_dataset.py

# Features
python scripts/build_features.py --tickers AAPL TSLA

# Training
python scripts/train_lstm.py                        # LSTM on price sequences
python scripts/train_lstm.py --epochs 100 --lr 0.0005   # with overrides
python scripts/train_nlp.py                         # NLP baseline on news headlines
python scripts/train_nlp.py --tickers AAPL TSLA     # specific tickers

# Validation
python scripts/validate_data.py
pytest tests/ -v                                    # 31 passing

# Reset everything
python scripts/reset_data.py                        # delete all generated data
python scripts/reset_data.py --keep-raw             # keep prices/news, reset features + models

# Browse the database
sqlite3 -header -column data/database.db "SELECT ticker, date, close FROM prices LIMIT 10;"
sqlite3 -header -column data/database.db "SELECT ticker, date, label_binary, label_return FROM labels LIMIT 10;"
```

### All scripts accept `--help`

```bash
python scripts/demo.py --help
python scripts/train_lstm.py --help
python scripts/collect_prices.py --help
```

---

## What Each File Does

### `src/config.py`
All settings in one place: file paths, ticker list (15 stocks), API key (from `.env`), prediction settings (binary, 4PM ET cutoff), model hyperparameters (placeholders for later tuning). Auto-creates `data/` directories on import.

### `src/database/schema.py`
Defines 5 SQLite tables. `prices` stores OHLCV data. `news` stores articles (unique by URL). `labels` stores up/down ground truth (populated by Tim's label generator). `predictions` stores model outputs. `run_log` tracks every script run for debugging.

### `src/data_collection/price_collector.py`
Downloads from Yahoo Finance with retry logic (3 attempts, exponential backoff) and deduplication (UNIQUE constraint on ticker+date). Logs every run.

### `src/data_collection/news_collector.py`
Fetches from Finnhub API with rate limiting (60 calls/min), word-boundary relevance filtering (checks headline + summary, avoids false positives for short tickers), and timestamp conversion to Eastern Time.

### `src/utils/api_clients.py`
Thin Finnhub wrapper. Handles rate limiting and has a cutoff filter (drops articles after 4PM ET).

### `src/data_processing/`
- **`label_generator.py`** — for each (ticker, date), compares today's close to the next trading day's close. Up = 1, down = 0. Stores label + percentage return in the `labels` table. Safe to re-run (skips duplicates).
- **`dataset_split.py`** — splits data chronologically (70/15/15). Saves to `data/processed/split_info.json`.
- **`standardization.py`** — date/timestamp conversion, 4PM ET cutoff rule. Skips weekends and fixed US market holidays automatically.
- **`price_validation.py`** — checks for missing data, >20% daily moves, zero volume, coverage gaps.
- **`news_validation.py`** — checks for missing fields, future timestamps, duplicate URLs, distribution.

### `src/features/`
- **`technical_indicators.py`** — computes RSI, MACD (line + signal + histogram), Bollinger Bands (upper/lower/width/position), and volume moving average from price data. Returns a DataFrame with 17 feature columns (OHLCV + daily return + indicators).
- **`sequence_generator.py`** — slides a 60-day window across the indicator data. Each window = one LSTM training sample. Features are min-max normalized per window so different price scales (e.g. $3 PLTR vs $250 AAPL) don't matter. Needs 60+ valid rows (after indicator warmup) to produce any output.
- **`text_features.py`** — extracts TF-IDF features from news headlines. For each (ticker, date) with a label, gathers headlines from the previous 3 days and converts them to a sparse feature vector. Supports separate fit/transform for proper train/val/test separation. Uses bigrams, English stop words, and configurable vocabulary size (default 5000).

### `src/models/`
- **`lstm_model.py`** — two-layer stacked LSTM for binary stock prediction. `StockLSTM` is the PyTorch module (input → LSTM-1 → dropout → LSTM-2 → dropout → linear → sigmoid). `LSTMTrainer` handles the training loop with early stopping on validation accuracy, evaluation with per-class recall, model save/load, and probability predictions. Configurable via `LSTM_CONFIG` in `config.py`.
- **`nlp_baseline.py`** — logistic regression on TF-IDF headline features. `NLPBaseline` wraps scikit-learn's `LogisticRegression` with balanced class weights. Uses `TextFeatureExtractor` for feature preparation. Saves/loads both the vectorizer and classifier together via joblib.

### `scripts/`
CLI entry points — all accept `--help`:

| Script | What it does |
|--------|-------------|
| `demo.py` | Full pipeline in one command. Accepts `--reset`, `--tickers`, `--all`, `--days`, `--skip-training`. |
| `reset_data.py` | Delete all generated data. `--keep-raw` preserves downloaded prices/news. |
| `setup_database.py` | Create the SQLite database and all 5 tables. |
| `collect_prices.py` | Download price data from Yahoo Finance. `--days`, `--tickers`. |
| `collect_news.py` | Fetch news from Finnhub API. `--days`, `--tickers`. |
| `collect_all_data.py` | Prices + news + validation in one step. `--days`. |
| `generate_labels.py` | Compute up/down labels from consecutive closing prices. |
| `split_dataset.py` | Chronological train/val/test split (70/15/15). |
| `build_features.py` | Technical indicators + LSTM sequences + date metadata. `--tickers`, `--seq-len`. |
| `validate_data.py` | Run data quality checks on prices and news. |
| `train_lstm.py` | Train the 2-layer LSTM. `--epochs`, `--batch-size`, `--lr`. |
| `train_nlp.py` | Train logistic regression on TF-IDF. `--tickers`, `--max-features`. |

### `tests/unit/`
31 tests covering schema, price collection, standardization, label generation, dataset splitting, technical indicators, and sequence generation. Run with `pytest tests/ -v`.

---

## Key Design Decisions

**SQLite over PostgreSQL** — zero setup, database is just a file. Sufficient for our data volume.

**Binary classification** — up/down is easier to evaluate than exact price. Random guessing = 50%, so >55% means the model learned something.

**4PM cutoff** — markets close at 4PM ET. News after close affects the *next* trading day, not today. Without this, models would train on future info (data leakage).

**Chronological splits** — time series must be split by date, not randomly. Train on past → predict future.

**250+ calendar days minimum** — the LSTM needs 60-day windows, and technical indicators consume ~34 days of warmup. With fewer than ~175 trading days, no sequences can be generated.

**Retry logic** — free APIs (Yahoo Finance, Finnhub) are flaky. Exponential backoff handles timeouts without crashing the pipeline.

**Run logging** — every collection run is recorded. When data looks wrong, check `SELECT * FROM run_log` to see what happened.

**Early stopping** — LSTM training stops when validation accuracy hasn't improved for 10 epochs, then restores the best weights. Prevents overfitting to training data.

**Balanced class weights** — the NLP baseline uses `class_weight="balanced"` so minor class imbalances (slightly more up than down days in a bull market) don't bias the model.

---

## Troubleshooting

**"No module named src"** — activate venv and run from project root.

**"Not enough data for sequences"** — you need ≥250 calendar days of price history. Run `python scripts/collect_prices.py --days 365` or use `python scripts/demo.py --days 365`.

**"sequence_dates.json not found"** — re-run `python scripts/build_features.py` (or `python scripts/demo.py --reset` for a full reset).

**"No data from yfinance"** — no data on weekends/holidays. Use `--days 365` for a wider range.

**"Finnhub API key is empty"** — add your key to `.env`: `NEWS_API_KEY=your_key`.

**"No training samples" for NLP** — Finnhub's free tier only returns articles from the last ~60 days. With 250+ days of prices, those articles fall in the validation/test period, not training. Fix: collect news regularly over time (e.g. run `python scripts/collect_news.py` weekly), or use `--days 120` for prices so the training window overlaps with available news.

**"Database locked"** — close other sqlite3 sessions. Reset: `python scripts/reset_data.py`.
