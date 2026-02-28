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

# Quick demo — collects data for 2 tickers, validates, prints samples
python scripts/demo.py

# Or step by step:
python scripts/setup_database.py
python scripts/collect_prices.py --tickers AAPL TSLA --days 90
python scripts/collect_news.py --tickers AAPL --days 7
python scripts/generate_labels.py
python scripts/split_dataset.py
python scripts/build_features.py --tickers AAPL TSLA
python scripts/validate_data.py

# Run tests (31 passing)
pytest tests/ -v

# Browse the database visually
# Install "SQLite Viewer" extension (alexcvzz.vscode-sqlite), then click data/database.db

# Or in terminal:
sqlite3 -header -column data/database.db "SELECT ticker, date, close FROM prices LIMIT 10;"
sqlite3 -header -column data/database.db "SELECT ticker, date, label_binary, label_return FROM labels LIMIT 10;"
```

---

## What Each File Does

### `src/config.py`
All settings in one place: file paths, ticker list (15 stocks), API key (from `.env`), prediction settings (binary, 4PM ET cutoff), model hyperparameters (placeholders for later tuning). Auto-creates `data/` directories on import.

### `src/database/schema.py`
Defines 5 SQLite tables. `prices` stores OHLCV data. `news` stores articles (unique by URL). `labels` stores up/down ground truth (populated by Tim's label generator). `predictions` stores model outputs (Week 5+). `run_log` tracks every script run for debugging.

### `src/data_collection/price_collector.py`
Downloads from Yahoo Finance with retry logic (3 attempts, exponential backoff) and deduplication (UNIQUE constraint on ticker+date). Logs every run.

### `src/data_collection/news_collector.py`
Fetches from Finnhub API with rate limiting (60 calls/min), relevance filtering (keeps articles mentioning the ticker/company), and timestamp conversion to Eastern Time.

### `src/utils/api_clients.py`
Thin Finnhub wrapper. Handles rate limiting and has a cutoff filter (drops articles after 4PM ET).

### `src/data_processing/`
- **`label_generator.py`** — for each (ticker, date), compares today's close to the next trading day's close. Up = 1, down = 0. Stores label + percentage return in the `labels` table. Safe to re-run (skips duplicates).
- **`dataset_split.py`** — splits data chronologically (70/15/15). Saves to `data/processed/split_info.json`.
- **`standardization.py`** — date/timestamp conversion, 4PM ET cutoff rule.
- **`price_validation.py`** — checks for missing data, >20% daily moves, zero volume, coverage gaps.
- **`news_validation.py`** — checks for missing fields, future timestamps, duplicate URLs, distribution.

### `src/features/`
- **`technical_indicators.py`** — computes RSI, MACD (line + signal + histogram), Bollinger Bands (upper/lower/width/position), and volume moving average from price data. Returns a DataFrame with 16 feature columns.
- **`sequence_generator.py`** — slides a 60-day window across the indicator data. Each window = one LSTM training sample. Features are min-max normalized per window so different price scales (e.g. $3 PLTR vs $250 AAPL) don't matter. Needs 60+ days of price data to produce any output.

### `scripts/`
CLI entry points: `setup_database.py`, `collect_prices.py`, `collect_news.py`, `collect_all_data.py`, `generate_labels.py`, `split_dataset.py`, `build_features.py`, `validate_data.py`, `demo.py`. All accept `--help`.

### `tests/unit/`
31 tests covering schema, price collection, standardization, label generation, dataset splitting, technical indicators, and sequence generation. Run with `pytest tests/ -v`.

---

## Key Design Decisions

**SQLite over PostgreSQL** — zero setup, database is just a file. Sufficient for our data volume.

**Binary classification** — up/down is easier to evaluate than exact price. Random guessing = 50%, so >55% means the model learned something.

**4PM cutoff** — markets close at 4PM ET. News after close affects the *next* trading day, not today. Without this, models would train on future info (data leakage).

**Chronological splits** — time series must be split by date, not randomly. Train on past → predict future.

**Retry logic** — free APIs (Yahoo Finance, Finnhub) are flaky. Exponential backoff handles timeouts without crashing the pipeline.

**Run logging** — every collection run is recorded. When data looks wrong, check `SELECT * FROM run_log` to see what happened.

---

## Troubleshooting

**"No module named src"** — activate venv and run from project root.

**"No data from yfinance"** — no data on weekends/holidays. Use `--days 30` for wider range.

**"Finnhub API key is empty"** — add your key to `.env`: `NEWS_API_KEY=your_key`.

**"Database locked"** — close other sqlite3 sessions. Reset: `rm data/database.db && python scripts/setup_database.py`.
