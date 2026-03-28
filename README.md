# News-to-Alpha

Predict next-day stock price movements (up/down) by combining an LSTM model on price history with an NLP model on news sentiment.

**15 tickers**: AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR

## Setup

```bash
git clone https://github.com/raphaelkaramagi/news-to-alpha.git
cd news-to-alpha

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env             # then add your Finnhub key to NEWS_API_KEY=
```

## Quick Start

The fastest way to test the entire system — one command runs the full pipeline
(collect prices → collect news → generate labels → split → build features →
train LSTM → train NLP):

```bash
python scripts/demo.py --reset          # fresh start, runs everything (all 15 tickers)
```

Customize:

```bash
python scripts/demo.py --tickers AAPL NVDA TSLA    # pick specific tickers
python scripts/demo.py --quick                      # fast test (AAPL + TSLA only)
python scripts/demo.py --days 365                   # more price history
python scripts/demo.py --skip-training              # data pipeline only, no models
```

## Step-by-Step Usage

If you prefer running each stage individually:

```bash
# 1. Data collection
python scripts/setup_database.py                         # create database (once)
python scripts/collect_prices.py --tickers AAPL TSLA     # defaults to 365 days
python scripts/collect_news.py --tickers AAPL --days 30  # news articles

# 2. Labels + splits
python scripts/generate_labels.py                        # up/down labels from prices
python scripts/split_dataset.py                          # chronological 70/15/15 split

# 3. Features
python scripts/build_features.py --tickers AAPL TSLA    # indicators + LSTM sequences

# 4. Training
python scripts/train_lstm.py                             # train LSTM model
python scripts/train_lstm.py --epochs 100 --lr 0.0005    # with overrides
python scripts/train_nlp.py                              # train NLP baseline
python scripts/train_nlp.py --tickers AAPL TSLA          # specific tickers

# 5. Validation
python scripts/validate_data.py                          # check data quality
pytest tests/ -v                                         # run unit tests (31 passing)
```

## Reset

Start fresh at any time:

```bash
python scripts/reset_data.py              # delete database + features + models
python scripts/reset_data.py --keep-raw   # keep downloaded data, reset features + models only
```

## Project Status

**Week 5 complete** — baseline LSTM and NLP models trainable on collected data.
See [docs/ROADMAP.md](docs/ROADMAP.md) for the full week-by-week plan, team assignments, and what's next.

## Structure

```
src/                    Core library (config, collectors, validation)
  data_collection/      Price + news collectors (working)
  data_processing/      Validation, standardization, labels, splits (working)
  database/             Schema definitions (working)
  features/             Technical indicators, LSTM sequences, TF-IDF (working)
  models/               LSTM model, NLP baseline (working); NLP advanced, ensemble (Week 6+)
  evaluation/           Metrics, backtesting (Week 6+)
  utils/                API clients (working)
scripts/                Runnable commands (collect, validate, train, demo, reset)
tests/                  Automated tests
data/                   Local database + data files (git-ignored, auto-created)
docs/                   Project overview, roadmap
```

## Viewing Data

Install the **SQLite Viewer** extension in VS Code/Cursor (`alexcvzz.vscode-sqlite`), then click `data/database.db` to browse tables visually. Or use `sqlite3 -header -column data/database.db` in the terminal.

## Docs

- **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** — detailed walkthrough of every file, how to test, design decisions
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — week-by-week plan with tasks per team member
