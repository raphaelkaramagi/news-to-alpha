# Project Overview

## What This Project Does

We predict whether a stock will go **up or down** the next trading day. Two independent models make predictions, and an ensemble combines them:

1. **Financial model** — learns patterns from 60-day windows of price data + technical indicators like RSI, MACD, Bollinger Bands
2. **News model (NLP)** — learns from recent headlines using TF-IDF (baseline) and later FinBERT (advanced)
3. **Ensemble** — weighted average of both, aiming to beat either one alone

The final product is a web app where you search a ticker and see a prediction with confidence score and the top headlines that influenced it.

---

## How to Test 

### Quick demo (2 minutes)

```bash
source .venv/bin/activate
python scripts/demo.py
```

This creates the database, downloads 2 weeks of AAPL + TSLA prices, optionally fetches news (if you have an API key in `.env`), validates quality, and prints sample rows.

### Test the price pipeline manually

```bash
# Fresh database
rm -f data/database.db
python scripts/setup_database.py

# Collect 3 tickers for 2 weeks
python scripts/collect_prices.py --tickers AAPL TSLA NVDA --days 14

# Look at what's in the database
sqlite3 data/database.db "SELECT ticker, date, close FROM prices ORDER BY ticker, date;"

# Run it again — should show 0 new rows (duplicates skipped)
python scripts/collect_prices.py --tickers AAPL TSLA NVDA --days 14
```

### Test the news pipeline

```bash
# Requires NEWS_API_KEY in .env
python scripts/collect_news.py --tickers AAPL --days 7

sqlite3 data/database.db "SELECT ticker, substr(title, 1, 50), published_at FROM news LIMIT 10;"
```

This takes a bit longer because of Finnhub's rate limit (60 calls/minute). For all 15 tickers it can take 3-5 minutes.

### Run the full pipeline

```bash
python scripts/collect_all_data.py
```

Does prices + news + validation in one go.

### Data quality check

```bash
python scripts/validate_data.py
```

Shows: coverage per ticker, missing data, price anomalies (>20% daily jumps), zero-volume days, article distribution.

### Run automated tests

```bash
pytest tests/ -v                           # all 13 tests
pytest tests/unit/test_schema.py -v        # just database tests
pytest tests/unit/test_standardization.py  # just date/time tests
pytest --cov=src tests/                    # with coverage report
```

### Explore the database directly

**VS Code / Cursor:** Install the **SQLite Viewer** extension (`alexcvzz.vscode-sqlite`). Then just click on `data/database.db` in the file explorer — it opens a visual table browser where you can click through tables, sort columns, and run queries without touching the terminal.

**Terminal:**

```bash
sqlite3 -header -column data/database.db
```

```sql
-- How much data do we have?
SELECT ticker, COUNT(*) AS days FROM prices GROUP BY ticker;
SELECT ticker, COUNT(*) AS articles FROM news GROUP BY ticker;

-- Recent prices
SELECT * FROM prices WHERE ticker='AAPL' ORDER BY date DESC LIMIT 5;

-- Run history
SELECT run_type, status, rows_added, started_at FROM run_log;

.quit
```

---

## File-by-File Breakdown

### `src/config.py`

Single source of truth for the whole project. Everything reads from here instead of hardcoding values. Contains:

- **File paths** — where the database, raw data, models live. All under `data/` which is git-ignored.
- **Ticker list** — the 15 stocks and a ticker-to-company-name mapping (used by the news relevance filter).
- **API key** — read from `.env` via `python-dotenv`. The key is never in code.
- **Prediction settings** — binary classification, 4 PM ET cutoff, Eastern timezone.
- **Data collection defaults** — 21-day lookback, 3 retries with 2-second exponential backoff.
- **Model hyperparameters** — LSTM config (60-day sequences, 2 layers of 50 units, 0.2 dropout) and NLP config (5000 TF-IDF features). These are placeholders that will be tuned in Weeks 6-7.

When config.py is imported, it auto-creates the `data/` directory tree if it doesn't exist.

### `src/database/schema.py`

Defines 5 SQLite tables. Uses `CREATE TABLE IF NOT EXISTS` so it's safe to run repeatedly.

**prices** — One row per (ticker, date). Stores OHLCV (Open, High, Low, Close, Volume) plus adjusted close. This is what the LSTM model will train on.

**news** — One row per unique URL. Stores the ticker it was fetched for, headline, source, publish time in ISO-8601 ET, and optional content/summary. The NLP model will train on titles from this table.

**labels** — Will be populated in Week 3. For each (ticker, date), stores whether the stock went up or down the next day (`label_binary`: 1=up, 0=down) and the percentage return. Both models will train against these labels.

**predictions** — Will be populated in Week 4+. Stores each model's output (probability, confidence) plus the actual outcome so we can calculate accuracy.

**run_log** — Every time a collection script runs, it logs: what tickers it tried, which succeeded/failed, how many rows were added, how long it took, and any error messages. Useful for debugging when something goes wrong.

All tables have indexes on (ticker, date) for fast lookups.

### `src/data_collection/base_collector.py`

A tiny abstract class. Both `PriceCollector` and `NewsCollector` inherit from it, which forces them to have the same `.collect()` interface. This means scripts like `collect_all_data.py` can call either collector the same way.

### `src/data_collection/price_collector.py`

Downloads stock prices from Yahoo Finance using the `yfinance` library.

Key details:
- **Retry logic** — if a download fails (network timeout, Yahoo being flaky), it retries up to 3 times with exponential backoff (2s, 4s, 8s). This is important because Yahoo Finance occasionally drops requests.
- **Duplicate prevention** — the prices table has a UNIQUE(ticker, date) constraint. If you run the collector twice for the same date range, the second run inserts 0 rows and logs the duplicates.
- **Multi-level column handling** — yfinance sometimes returns DataFrames with multi-level column headers (Ticker, Metric). The code detects this and flattens it.
- **Run logging** — after every collection, writes a summary to `run_log` with timing, counts, and any errors.

### `src/data_collection/news_collector.py`

Fetches company news from Finnhub's REST API.

Key details:
- **Rate limiting** — Finnhub's free tier allows 60 API calls per minute. The `FinnhubClient` wrapper tracks calls and auto-sleeps when approaching the limit. Without this, you'd get 429 errors halfway through collecting 15 tickers.
- **Relevance filter** — Finnhub returns articles that are loosely related to a company. Many don't actually mention it. The filter checks if the ticker symbol or company name appears in the headline. If this is too aggressive (keeps less than 10%), it falls back to keeping everything. This is a simple heuristic that will be improved in Week 3.
- **Timestamp handling** — Finnhub returns Unix timestamps in UTC. The collector converts them to Eastern Time ISO-8601 strings (e.g., `2026-02-07T14:13:00-05:00`) matching the standard in `PROJECT_SPEC.md`.
- **Deduplication** — uses the article URL as the unique key. Re-running won't create duplicates.

### `src/utils/api_clients.py`

Thin wrapper around Finnhub's REST API. Separated from the collector so the rate-limiting logic is reusable. Also has a `filter_by_cutoff()` helper that drops articles published after market close (4 PM ET).

### `src/data_processing/price_validation.py`

Runs 4 quality checks on price data:

1. **Missing values** — any rows where OHLCV fields are NULL (shouldn't happen with Yahoo Finance, but good to check).
2. **Price anomalies** — daily close-to-close moves greater than 20%. These are either real (earnings surprises) or data errors. Uses a SQL window function (`LAG`) to compute day-over-day change.
3. **Zero volume** — trading days with 0 volume. Can indicate a data problem or a market holiday that shouldn't be in the data.
4. **Coverage** — how many trading days we have per ticker and the date range. Useful for spotting tickers where data collection silently failed.

### `src/data_processing/news_validation.py`

Runs 4 quality checks on news data:

1. **Missing fields** — articles without title, source, or published_at.
2. **Future timestamps** — articles with publish dates in the future (data error from the API).
3. **Duplicate URLs** — should never happen because of the UNIQUE constraint, but checks anyway.
4. **Distribution** — article count per ticker. Some tickers (small-cap like DAL, MAR) get much less news coverage than AAPL or TSLA. This will affect model performance and is something to account for.

### `src/data_processing/standardization.py`

Utility class with three static methods:

- `standardize_date()` — converts any date format ("February 7, 2026", "2026-02-07", datetime objects) to `YYYY-MM-DD`.
- `standardize_timestamp()` — converts Unix timestamps or strings to ISO-8601 with timezone.
- `apply_cutoff_rule()` — the core business logic for news timing. News published before 4 PM ET on day T is used to predict day T+1. News published after 4 PM is used to predict day T+2. This prevents data leakage (using information that wasn't available when the market was open). Weekend/holiday handling is deferred to Week 3.

### `scripts/`

Six entry points:

- `setup_database.py` — run once to create all 5 tables. Safe to run again.
- `collect_prices.py` — accepts `--tickers` and `--days` flags. Defaults to all 15 tickers, 21 days.
- `collect_news.py` — same flags. Requires `NEWS_API_KEY` in `.env`.
- `collect_all_data.py` — runs both collectors then validation. Best for weekly data refreshes.
- `validate_data.py` — prints a quality report without collecting anything new.
- `demo.py` — end-to-end demo: creates DB, collects 2 tickers, validates, prints sample data.

All scripts add the project root to `sys.path` so `from src.config import ...` works without installing the package.

### `tests/unit/`

**test_schema.py** (4 tests)
- All 5 tables get created
- Running create twice doesn't break anything
- Inserting duplicate (ticker, date) in prices throws IntegrityError
- Inserting duplicate URL in news throws IntegrityError

**test_price_collector.py** (4 tests)
- Collecting AAPL for a real date range succeeds and inserts rows
- An invalid ticker fails gracefully (no crash, recorded as failure)
- Running collection twice: second run adds 0 rows, duplicates counted
- Every run writes to run_log table

**test_standardization.py** (5 tests)
- Parses "February 7, 2026" → "2026-02-07"
- Parses "2026-02-07" → "2026-02-07"
- Converts Unix timestamp to ISO-8601 with ET offset
- Article at 2 PM ET → predicts next day
- Article at 5 PM ET → predicts day after next

The price collector tests hit the real Yahoo Finance API (they need network). Schema and standardization tests are pure offline.

---

## Key Design Decisions

### Why SQLite instead of PostgreSQL?
SQLite requires zero setup — no server, no Docker, no config. The database is just a file in `data/`. For a project with 15 tickers and a few months of data, SQLite is more than sufficient. If we needed concurrent writes or millions of rows, we'd switch to PostgreSQL, but the schema is designed to be portable.

### Why binary classification instead of regression?
Predicting whether a stock goes up or down (binary) is easier to evaluate than predicting exact price. A baseline of random guessing gives 50% accuracy, so anything above 55% is meaningful. Regression would need MSE/RMSE and it's harder to tell if results are useful.

### Why the 4 PM cutoff?
US stock markets close at 4:00 PM Eastern. News published after market close can't affect that day's price — it affects the next trading day. Without this rule, the model would train on "future" information (data leakage), making accuracy look artificially high but fail in real use.

### Why both LSTM and NLP?
Price patterns and news sentiment capture different signals. An LSTM can learn that "RSI below 30 followed by volume spike" often precedes a bounce. An NLP model can learn that "FDA approves" headlines are bullish for pharma stocks. Neither alone captures both, and the ensemble should outperform either one.

### Why retry logic?
Yahoo Finance and Finnhub are free-tier APIs. They occasionally time out, return empty responses, or throttle requests. Without retries, a single flaky request would cause data gaps. Exponential backoff (wait 2s, then 4s, then 8s) is standard practice.

### Why log every run?
The `run_log` table records every collection attempt. When someone says "I ran the script but TSLA data is missing," you can check `SELECT * FROM run_log` to see if the run succeeded, how many rows were added, and what errors occurred. This saves hours of debugging.

---

## What's Coming Next

See [ROADMAP.md](ROADMAP.md) for the full week-by-week plan with specific tasks per team member, which files to create, and what integrates with what.

**Short version:** Week 3 generates labels and splits the data. Weeks 4–5 build baseline models. Weeks 6–7 improve them. Weeks 8–9 combine them. Week 10 polishes. Week 11 is the web demo.

---

## Troubleshooting

**"No module named src"** — make sure you're in the project root and the venv is activated. Scripts handle this with `sys.path.insert`, but if running code interactively, do `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`.

**"No data returned from yfinance"** — Yahoo Finance has no data for weekends/holidays. Try `--days 30` for a wider window. Also check your internet connection.

**"Finnhub API key is empty"** — edit `.env` and add your key: `NEWS_API_KEY=your_key_here`. Get a free key at https://finnhub.io/register.

**"Database locked"** — another script or sqlite3 session is holding a connection. Close everything and retry. Nuclear option: `rm data/database.db && python scripts/setup_database.py`.

**"0 articles for ticker X"** — some tickers get sparse news coverage on Finnhub's free tier, especially smaller companies like DAL or MAR. This is expected and will be accounted for in the NLP model.
