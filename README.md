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

## Usage

```bash
python scripts/setup_database.py                        # create database (once)
python scripts/collect_prices.py --tickers AAPL TSLA    # get price data
python scripts/collect_news.py --tickers AAPL           # get news articles
python scripts/generate_labels.py                       # generate up/down labels
python scripts/split_dataset.py                         # chronological train/val/test split
python scripts/build_features.py --tickers AAPL TSLA   # indicators + LSTM sequences
python scripts/validate_data.py                         # check data quality
python scripts/demo.py                                  # quick end-to-end demo
pytest tests/ -v                                        # run tests (31 passing)
```

## Project Status

**Week 4 complete** — labels, splits, technical indicators, and LSTM sequences done.
See [docs/ROADMAP.md](docs/ROADMAP.md) for the full week-by-week plan, team assignments, and what's next.

## Structure

```
src/                    Core library (config, collectors, validation)
  data_collection/      Price + news collectors (working)
  data_processing/      Validation, standardization, labels, splits (working)
  database/             Schema definitions (working)
  features/             Technical indicators, LSTM sequences (working)
  models/               LSTM, NLP baseline, NLP advanced, ensemble (Week 5+)
  evaluation/           Metrics, backtesting (Week 6+)
  utils/                API clients (working)
scripts/                Runnable commands (collect, validate, demo)
tests/                  Automated tests
data/                   Local database + data files (git-ignored, auto-created)
docs/                   Project overview, roadmap, git guide
```

Folders like `features/`, `models/`, and `evaluation/` have empty `__init__.py` files — they're placeholders for upcoming weeks. The roadmap explains exactly what goes in each one.

## Viewing Data

Install the **SQLite Viewer** extension in VS Code/Cursor (`alexcvzz.vscode-sqlite`), then click `data/database.db` to browse tables visually. Or use `sqlite3 -header -column data/database.db` in the terminal.

## Docs

- **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** — detailed walkthrough of every file, how to test, design decisions
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — week-by-week plan with tasks per team member
- **[docs/GIT_GUIDE.md](docs/GIT_GUIDE.md)** — branching, PRs, and team workflow
