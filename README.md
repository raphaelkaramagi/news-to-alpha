# News-to-Alpha

Predict whether a stock will go **up or down** the next trading day by combining two signals: price history (LSTM) and news headlines (NLP). The end goal is a web app where you search a ticker and see a prediction with confidence and the headlines behind it.

**15 tickers tracked**: AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR

---

## Setup

```bash
git clone https://github.com/raphaelkaramagi/news-to-alpha.git
cd news-to-alpha

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env             # add your Finnhub API key: NEWS_API_KEY=your_key
```

---

## Quick Start

One command runs the entire pipeline — collects data, builds features, trains both models:

```bash
python scripts/demo.py --reset      # wipe and run fresh (all 15 tickers)
python scripts/demo.py              # re-run, skips already-collected data
```

Customize:

```bash
python scripts/demo.py --quick                      # fast 2-ticker test (AAPL + TSLA)
python scripts/demo.py --tickers AAPL NVDA TSLA     # pick specific tickers
python scripts/demo.py --days 365                   # more price history
python scripts/demo.py --skip-training              # data pipeline only, no model training
```

---

## Step-by-Step Usage

If you prefer running each stage individually:

```bash
# 1. Database
python scripts/setup_database.py                          # create tables (once)

# 2. Data collection
python scripts/collect_prices.py                          # all 15 tickers, 365 days
python scripts/collect_news.py                            # all 15 tickers, latest articles
python scripts/collect_all_data.py                        # prices + news + validation in one go

# 3. Labels + splits
python scripts/generate_labels.py                         # up/down label per (ticker, date)
python scripts/split_dataset.py                           # chronological 70/15/15 split

# 4. Features
python scripts/build_features.py                          # indicators + LSTM sequences

# 5. Training
python scripts/train_lstm.py                              # LSTM on price sequences
python scripts/train_lstm.py --epochs 100 --lr 0.0003     # with overrides
python scripts/train_nlp.py                               # NLP baseline on news headlines

# 6. Validation
python scripts/validate_data.py                           # data quality checks
pytest tests/ -v                                          # unit tests
```

All scripts accept `--help` for full options.

---

## Reset

The database (with collected news) is preserved by default so accumulated data isn't lost:

```bash
python scripts/reset_data.py          # reset features + models, keep database
python scripts/reset_data.py --full   # delete everything including database
```

---

## Project Status

**Week 5 complete.** Both baseline models train and predict above the 50% random baseline:

| Model | What it uses | Test accuracy |
|-------|-------------|---------------|
| LSTM | 60-day price + indicator sequences | ~51% |
| NLP baseline | TF-IDF headlines → logistic regression | ~53% |

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full plan and what's coming in Weeks 6–11.

---

## Project Structure

```
src/
  config.py               central settings (tickers, paths, model hyperparams)
  database/               SQLite schema (5 tables: prices, news, labels, predictions, run_log)
  data_collection/        price collector (Yahoo Finance), news collector (Finnhub)
  data_processing/        validation, standardization, label generation, dataset splitting
  features/               technical indicators, LSTM sequence builder, TF-IDF text features
  models/                 LSTM model (PyTorch), NLP baseline (scikit-learn)
  evaluation/             [Week 6+] metrics, backtesting
  utils/                  API clients

scripts/                  runnable CLI commands (all accept --help)
tests/unit/               automated tests (pytest)
data/                     git-ignored — database, features, trained models (auto-created)
docs/                     project overview, roadmap
```

---

## Viewing the Data

Install **SQLite Viewer** in VS Code/Cursor (`alexcvzz.vscode-sqlite`) and click `data/database.db` to browse tables. Or from the terminal:

```bash
sqlite3 -header -column data/database.db "SELECT ticker, date, close FROM prices LIMIT 10;"
sqlite3 -header -column data/database.db "SELECT ticker, date, label_binary, label_return FROM labels LIMIT 10;"
```

---

## Docs

- **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** — detailed file-by-file walkthrough, design decisions, troubleshooting
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — week-by-week plan with team assignments
