# News-to-Alpha

Predict whether a stock will go up or down the next trading day by combining three signals: price history (LSTM), news headlines (TF-IDF + logistic regression), and sentence-embedding news (MiniLM + logistic regression). A learned meta-model stacks the three into a calibrated ensemble, and a Flask web app lets you search any ticker, switch between models, and retrain on demand.

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

The fastest path is the single-command orchestrator. It wraps every step
(collect → labels → split → LSTM → NLP → embeddings → ensemble → evaluate)
and accepts a **preset** plus optional overrides:

```bash
python scripts/run_pipeline.py --preset quick        # 2 tickers, 1-day horizon, 1 seed
python scripts/run_pipeline.py --preset balanced     # 5 tickers, 2y, 3-day horizon, 3 seeds
python scripts/run_pipeline.py --preset advanced     # full universe + FinBERT
```

Dry-run to inspect the resolved config without executing:

```bash
python scripts/run_pipeline.py --preset balanced --dry-run
```

Common overrides:

```bash
python scripts/run_pipeline.py --preset balanced --tickers AAPL NVDA --horizon 3 --min-move-pct 0.5
python scripts/run_pipeline.py --preset balanced --skip-collect --skip-news   # re-train without redownloading
python scripts/run_pipeline.py --preset balanced --use-finbert                # FinBERT sentiment in NLP
```

The older `scripts/demo.py` still works for the simpler legacy flow, but
`run_pipeline.py` is what the Flask app uses under the hood.

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
python scripts/train_lstm.py --horizon 3 --seeds 42 1337 2024      # seed-ensemble LSTM
python scripts/train_lstm.py --horizon 3 --min-move-pct 0.5        # drop-flat filter on train
python scripts/train_nlp.py --horizon 3 --min-move-pct 0.5         # TF-IDF + publisher + content snippet
python scripts/train_news_embeddings.py --horizon 3 --use-finbert  # MiniLM / FinBERT + relevance-weighted pooling
python scripts/build_ensemble.py                                    # HistGradientBoosting meta-model + calibration

# 6. Evaluation
python scripts/evaluate_predictions.py --horizon 3        # accuracy/F1/AUC + `evaluation_by_confidence.csv`
python scripts/validate_data.py                           # data quality checks
pytest tests/ -v                                          # unit tests

# 7. Web app
python app/server.py                                      # http://localhost:5000
```

The app serves three routes:

| Route        | Purpose                                                       |
|--------------|---------------------------------------------------------------|
| `/`          | Landing / configure page — pick a preset and run the pipeline |
| `/dashboard` | Interactive dashboard (see below)                             |
| `/admin`     | Per-model retrain / reset controls                            |

The header on every page links **Home** (`/`) ↔ **Admin** (`/admin`), and the
`Reconfigure` button opens a drawer to re-run the pipeline without leaving the
dashboard.

The dashboard is organised top-to-bottom as:

1. **Hero row** — ticker + model selectors, the latest call, and a
   "Last resolved call" card with a 7-dot hit/miss history strip.
2. **Price & prediction chart** with a vertical date marker driven by the
   rewind slider below it. An "all-up" chip appears above the chart on dates
   where every ticker is predicted bullish.
3. **Rewind slider** — moves the chart marker and updates the latest-call card,
   headlines, rationale, and all-up chip.
4. **Tabs** — `Headlines`, `Why this call`, `Accuracy & history`, and
   `Reliability`.
   - The Reliability tab plots calibration (predicted confidence vs empirical
     accuracy) and a histogram of the ensemble's predicted probabilities so
     plateau bands are visible at a glance.

Confidence in the latest-call card now shows the *empirical* win-rate at that
confidence level (sourced from `evaluation_by_confidence.csv`) alongside the
`|p-0.5|*2` score.

Keyboard shortcuts: `←`/`→` cycle tickers, `1..4` switch model
(ensemble / lstm / tfidf / embeddings), `r` opens the reconfigure drawer,
`Esc` closes it.

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

**Week 10 complete.** All three base models train end-to-end with probability
calibration; a `HistGradientBoosting` meta-model stacks them on 10 features and
gets its own isotonic + temperature calibration. The Flask app covers both the
configure-and-run flow and a rich interactive dashboard:

| Model               | What it uses                                                            |
|---------------------|-------------------------------------------------------------------------|
| LSTM (`lstm_price`) | 60-day scaled price + ~20 indicators + ticker embed, **seed ensemble + isotonic calibration + BCE pos_weight**  |
| TF-IDF              | TF-IDF bigrams over headlines **+ content snippet**, publisher one-hot, **Finnhub sentiment/relevance** |
| Embeddings          | MiniLM / optional FinBERT embeddings **relevance-weighted pooled**, + publisher + side features |
| Ensemble            | **HistGradientBoosting** meta-model over 10 features (base probas, confidences, `has_news`, `n_headlines`, `spy_return_5d`, `all_agree`) + adaptive calibration (isotonic above ≥2k val rows, else sigmoid) + temperature scaling |

New knobs you can tune end-to-end:

- `--horizon {1,3}` — 3-day direction labels usually give higher accuracy and
  sharper confidence than next-day.
- `--min-move-pct X` — drop training rows with `|return_horizon| < X` so the
  models stop learning from near-flat days.
- `--seeds 42 1337 2024` — train multiple LSTMs and average their probas.
- `--use-finbert` — swap MiniLM for FinBERT sentiment embeddings.
- `evaluation_by_confidence.csv` — accuracy per confidence decile, so you can
  see the high-conviction subset of calls.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full timeline and [docs/DEPLOY.md](docs/DEPLOY.md) for deploying the app to Railway/Render.

---

## Project Structure

```
src/
  config.py               central settings (tickers, paths, model hyperparams)
  database/               SQLite schema (5 tables: prices, news, labels, predictions, run_log)
  data_collection/        price collector (Yahoo Finance), news collector (Finnhub)
  data_processing/        validation, standardization, label generation, dataset splitting
  features/               technical indicators, LSTM sequence builder, news sentiment
  models/                 LSTM (PyTorch) + shared news pipeline (cutoff-aligned dataset)
  utils/                  API clients

scripts/                  runnable CLI commands (all accept --help)
app/                      Flask web app (server.py, jobs.py, templates/)
tests/unit/               pytest tests/ -v
data/                     git-ignored — database, features, trained models (auto-created)
docs/                     project overview, roadmap, deploy guide
```

---

## Viewing the Data

Install **SQLite Viewer** in VS Code (`alexcvzz.vscode-sqlite`) and click `data/database.db` to browse tables. Or from the terminal:

```bash
sqlite3 -header -column data/database.db "SELECT ticker, date, close FROM prices LIMIT 10;"
sqlite3 -header -column data/database.db "SELECT ticker, date, label_binary, label_return FROM labels LIMIT 10;"
```

---

## Docs

- **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** — detailed file-by-file walkthrough, design decisions, troubleshooting
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — week-by-week plan with team assignments
