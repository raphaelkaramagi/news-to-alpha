# Project Overview

## What It Does

Predicts whether a stock goes **up or down** the next trading day using two independent models:

1. **LSTM** — looks at 60 days of price history + technical indicators (RSI, MACD, Bollinger Bands, etc.) to learn price momentum patterns.
2. **NLP** — reads recent news headlines for each ticker and extracts text features to capture market sentiment.
3. **Ensemble** *(Week 8+)* — combines both model outputs into a single, more confident prediction.

End goal: a web app where you search a ticker and see tomorrow's predicted direction with a confidence score and the headlines that influenced it.

---

## How to Run

```bash
source .venv/bin/activate
```

### Easiest: full pipeline in one command

Runs everything — data collection, labels, feature engineering, and model training:

```bash
python scripts/demo.py --reset          # wipe and run fresh
python scripts/demo.py                  # re-run, skips already-collected data
python scripts/demo.py --skip-training  # data pipeline only, no model training
```

Customize scope:

```bash
python scripts/demo.py --quick                      # fast 2-ticker test (AAPL + TSLA)
python scripts/demo.py --tickers AAPL NVDA TSLA     # specific tickers
python scripts/demo.py --days 365                   # more price history
```

### Step by step (more control)

```bash
# Data
python scripts/setup_database.py                     # create tables (once)
python scripts/collect_prices.py                     # all tickers, 365 days
python scripts/collect_news.py                       # latest headlines (run weekly to accumulate)
python scripts/collect_all_data.py                   # prices + news + validation in one step

# Labels + split
python scripts/generate_labels.py                    # up/down label per (ticker, date)
python scripts/split_dataset.py                      # chronological 70/15/15 split

# Features
python scripts/build_features.py                     # compute indicators, build LSTM sequences

# Training
python scripts/train_lstm.py                         # price-based LSTM
python scripts/train_lstm.py --epochs 100 --lr 0.0003
python scripts/train_nlp.py                          # news-based NLP baseline

# Checks
python scripts/validate_data.py                      # data quality report
pytest tests/ -v                                     # unit tests

# Reset
python scripts/reset_data.py                         # clear features + models (keeps database)
python scripts/reset_data.py --full                  # clear everything
```

All scripts accept `--help`.

---

## What Each File Does

### `src/config.py`
The single source of truth for the whole project. Defines file paths, the 15 tickers, the Finnhub API key (read from `.env`), the 4PM ET prediction cutoff, and LSTM/NLP hyperparameters. Also auto-creates `data/` subdirectories on import so nothing else has to.

### `src/database/schema.py`
Creates and manages 5 SQLite tables:
- `prices` — daily OHLCV data per ticker
- `news` — headline + summary + URL per article (deduplicated by URL)
- `labels` — binary up/down label and percentage return per (ticker, date)
- `predictions` — model output scores stored for comparison and backtesting
- `run_log` — records every script execution for debugging data issues

### `src/data_collection/price_collector.py`
Downloads historical OHLCV data from Yahoo Finance. Handles retries with exponential backoff (APIs are flaky), deduplicates on insert, and logs each collection run.

### `src/data_collection/news_collector.py`
Fetches company news from Finnhub. Rate-limits to 60 calls/minute (free tier limit), applies a relevance filter using word-boundary regex (avoids false matches like "GS" in "things"), and converts timestamps to Eastern Time. Free tier returns roughly the last 21 days of articles — run this weekly to build up training data over time.

### `src/utils/api_clients.py`
Thin wrapper around the Finnhub REST API. Handles HTTP calls and exposes `get_company_news(ticker, from_date, to_date)`.

### `src/data_processing/label_generator.py`
For each (ticker, date) pair, compares today's closing price to the next trading day's closing price. If the next day is higher → label = 1 (up), otherwise → label = 0 (down). Stores the label and percentage return. Safe to re-run — skips already-labeled dates.

### `src/data_processing/dataset_split.py`
Splits all dates chronologically into train (70%) / val (15%) / test (15%). Saves the date lists to `data/processed/split_info.json` so every training script uses the same split. Never shuffles — time series data must always be split in order to avoid data leakage.

### `src/data_processing/standardization.py`
Handles date and timestamp normalization. Implements the 4PM ET cutoff rule: news published before 4PM is assigned to that trading day's prediction; news published after 4PM or on a weekend/holiday is assigned to the next valid trading day.

### `src/data_processing/price_validation.py`
Scans the price table for data quality issues: missing days, suspicious single-day moves (>20%), zero volume, and coverage gaps.

### `src/data_processing/news_validation.py`
Checks the news table for missing fields, future-dated timestamps, duplicate URLs, and coverage per ticker.

### `src/features/technical_indicators.py`
Computes 17 features from raw OHLCV data:
- **RSI** — is the stock overbought (>70) or oversold (<30)?
- **MACD** (line, signal, histogram) — is momentum building or fading?
- **Bollinger Bands** (upper, lower, middle, width, position) — is price near the top or bottom of recent range?
- **Volume MA + ratio** — is trading volume unusually high or low?
- **Daily return** — percentage price change day-over-day (direct directional signal)

### `src/features/sequence_generator.py`
Slides a 60-day window across each ticker's indicator data. Each window becomes one LSTM training sample. Features are min-max normalized per window so stocks with very different price levels (e.g. $3 PLTR vs $250 AAPL) are comparable. Requires at least 60 valid rows after indicator warmup (~34 days for MACD), so you need roughly 250+ calendar days of price data.

### `src/features/text_features.py`
Converts news headlines into numerical features using TF-IDF. For each labeled (ticker, date), gathers all headlines from the previous 3 days and joins them into a text document. Uses bigrams, removes English stop words, and caps vocabulary at a configurable size (default 5000). Supports separate `fit` (on training data only) and `transform` (on val/test) to prevent data leakage.

### `src/models/lstm_model.py`
Two-layer stacked LSTM for binary price direction prediction.
- `StockLSTM` (PyTorch): input (batch, 60, 17) → LSTM-1 (64 units) → dropout → LSTM-2 (64 units) → dropout → linear → sigmoid → P(up)
- `LSTMTrainer`: training loop with early stopping, LR scheduler (halves LR when val accuracy plateaus), gradient clipping, and per-class recall reporting. Saves/loads full checkpoints including config and training history.

### `src/models/nlp_baseline.py`
Logistic regression on TF-IDF headline features. Intentionally simple — it's a baseline to establish whether news carries any predictive signal before investing in heavier models (FinBERT, embeddings). Uses balanced class weights to handle class imbalance. Saves the vectorizer and classifier together so they can be loaded as a unit for inference.

### `src/evaluation/`
Empty placeholder for Week 6. Will contain `metrics.py` (confusion matrix, AUC-ROC, F1, precision/recall curves) and `backtester.py` (simulated trading returns from model predictions).

### `scripts/`

All scripts accept `--help`.

| Script | What it does |
|--------|-------------|
| `demo.py` | End-to-end pipeline. Options: `--reset`, `--quick`, `--tickers`, `--days`, `--skip-training`. Default runs all 15 tickers. |
| `reset_data.py` | Clear processed features and trained models. Database is kept by default (preserving accumulated news). `--full` deletes everything. |
| `setup_database.py` | Create the SQLite database and all 5 tables. Run once. |
| `collect_prices.py` | Download OHLCV data from Yahoo Finance. `--tickers`, `--days`. |
| `collect_news.py` | Fetch headlines from Finnhub. `--tickers`, `--days`. Run weekly to accumulate news. |
| `collect_all_data.py` | Prices + news + validation in one step. `--days`. |
| `generate_labels.py` | Compute up/down labels from price data. Safe to re-run. |
| `split_dataset.py` | Chronological 70/15/15 split. Saves to `data/processed/split_info.json`. |
| `build_features.py` | Compute technical indicators and generate 60-day LSTM sequences. `--tickers`. |
| `validate_data.py` | Run price and news data quality checks. |
| `train_lstm.py` | Train the LSTM on price sequences. `--epochs`, `--batch-size`, `--lr`. |
| `train_nlp.py` | Train the NLP baseline on headlines. `--tickers`, `--max-features`. |

### `tests/unit/`
Automated tests covering schema creation, price collection, standardization/cutoff logic, label generation, dataset splitting, technical indicator computation, and sequence generation. Run with `pytest tests/ -v`.

---

## Key Design Decisions

**SQLite over PostgreSQL** — zero setup, the database is just a file in `data/`. More than adequate for our data volume, and anyone can inspect it with the SQLite Viewer extension.

**Binary classification (up/down)** — simpler to evaluate than predicting exact prices. A model that beats 50% is learning something real. A useful target is >55% sustained on the test set.

**4PM ET cutoff rule** — stock markets close at 4PM Eastern. Any news published after 4PM affects the *next* trading day's price, not today's. Without this rule, the model would train on future information (data leakage), making accuracy appear artificially high and failing completely in production.

**Chronological splits** — for time series, you must train on older data and test on newer data. Random shuffling would leak future data into training and give misleadingly high accuracy.

**LSTM needs 250+ calendar days** — technical indicators take ~34 days to warm up (MACD uses a 26-day EMA), and the LSTM needs 60-day windows. Without enough history, no training sequences can be generated.

**NLP split is independent of the LSTM split** — free-tier news APIs only return roughly the last 21 days of articles. This is far shorter than the LSTM's 250-day price window, so sharing a single date split would leave the NLP model with zero training samples. The NLP model builds its own chronological 70/15/15 split from whatever news dates are available.

**Database preserved on reset** — news articles take time to accumulate (free APIs only return recent articles). Resetting features or models should not wipe collected news. `reset_data.py` preserves the database by default; use `--full` to clear everything.

**Retry logic with backoff** — free APIs (Yahoo Finance, Finnhub) are unreliable. Exponential backoff prevents a single timeout from crashing a long collection run.

**Run logging** — every script execution is recorded in the `run_log` table with timestamps and stats. When data looks wrong, `SELECT * FROM run_log ORDER BY started_at DESC LIMIT 20;` shows exactly what ran and when.

**Gradient clipping + orthogonal init** — LSTMs are prone to exploding/vanishing gradients. Clipping gradients to norm 1.0 and initializing recurrent weights orthogonally stabilizes training and prevents the model from collapsing to "predict everything the same class."

---

## Troubleshooting

**"No module named src"** — you're not in the project root, or the venv isn't active. Run `source .venv/bin/activate` from the `news-to-alpha/` directory.

**"Not enough data for sequences"** — need ≥250 calendar days of price history. Run `python scripts/demo.py --days 365` or `python scripts/collect_prices.py --days 365` then rebuild features.

**"sequence_dates.json not found"** — re-run `python scripts/build_features.py` to regenerate the feature files.

**"No data from yfinance"** — Yahoo Finance returns no data for weekends/holidays. Use `--days 365` to ensure enough trading days are covered.

**"Finnhub API key is empty"** — copy `.env.example` to `.env` and add your key: `NEWS_API_KEY=your_key_here`.

**NLP accuracy is low or erratic** — the NLP model trains on very few samples because free-tier news APIs only return ~21 days of articles. Run `python scripts/collect_news.py` weekly to build up more training data over time. More data = better calibration.

**"Database locked"** — another process has the database open. Close any SQLite browser sessions or terminals using the database, then retry.
