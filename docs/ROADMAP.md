# Project Roadmap

Week-by-week plan & team assignments

---

## Team & Roles

| People   | Focus area                         | Review partner |
|----------|------------------------------------|----------------|
| Raphael | Price data, technical indicators, LSTM | Tim        |
| Tim      | Database, sequence generation, LSTM training | Raphael |
| Moses    | News collection, NLP baseline, NLP advanced | Gordon  |
| Gordon   | API clients, NLP preprocessing, embeddings | Moses   |
| Cheri    | Project structure, standardization, ensemble, integration | cross-team |

---

## Where Things Live

```
news-to-alpha/
├── src/                        # all reusable code lives here
│   ├── config.py               
│   ├── database/
│   │   └── schema.py           # table definitions (5 tables)
│   ├── data_collection/
│   │   ├── base_collector.py   # abstract interface both collectors share
│   │   ├── price_collector.py  # Yahoo Finance → prices table
│   │   └── news_collector.py   # Finnhub API → news table
│   ├── data_processing/
│   │   ├── standardization.py  # date/time utils, cutoff rule
│   │   ├── price_validation.py # quality checks on price data
│   │   └── news_validation.py  # quality checks on news data
│   ├── utils/
│   │   └── api_clients.py      # Finnhub rate-limited wrapper
│   ├── features/               # EMPTY — Week 4: technical indicators, sequences, text features
│   ├── models/                 # EMPTY — Week 5+: LSTM, NLP baseline, NLP advanced, ensemble
│   └── evaluation/             # EMPTY — Week 6+: metrics, backtesting, comparison tools
├── scripts/                    # runnable entry points (python scripts/xxx.py)
├── tests/unit/                 # pytest tests
├── docs/                       # project docs, roadmap, git guide
├── data/                       # git-ignored, created automatically by config.py
│   ├── raw/prices/             # intermediate CSVs if needed
│   ├── raw/news/               # intermediate JSONs if needed
│   ├── processed/              # cleaned/split datasets (Week 3+)
│   ├── models/                 # saved model weights (Week 5+)
│   └── database.db             # SQLite — all collected data
└── app/                        # doesn't exist yet — Week 11 web demo
```

---

## Week 1 — Project Setup ✓

**Status: DONE**

What was built:
- Repository structure, `.gitignore`, `requirements.txt`
- `src/config.py` — centralized settings, paths, ticker list, API key loading
- `src/database/schema.py` — 5 SQLite tables (prices, news, labels, predictions, run_log)
- `scripts/setup_database.py` — creates the database
- Unit tests for schema creation and constraints

---

## Week 2 — Data Collection & Validation ✓

**Status: DONE**

What was built:
- Price collector with retry logic and deduplication
- News collector with Finnhub rate limiting and relevance filtering
- Validation modules for both pipelines
- Date/time standardization with 4 PM ET cutoff rule
- 6 scripts (collect prices, collect news, collect all, validate, setup DB, demo)
- 13 unit tests, all passing

---

## Week 3 — Labels & Dataset Preparation

**Goal:** Turn raw price data into training labels, link news articles to the right prediction dates, and split everything into train/validation/test sets.

### Tasks by person

**Tim** — Label generation
- Create `src/data_processing/label_generator.py`
- For each (ticker, date) row in `prices`, compute whether the stock went up or down the *next* trading day
- Store `label_binary` (1 = up, 0 = down) and `pct_return` in the `labels` table
- Handle edge cases: last day has no next-day return, weekends/holidays should be skipped
- Create `scripts/generate_labels.py` to run this from command line
- Write tests in `tests/unit/test_label_generator.py`
- **Integrates with:** `src/database/schema.py` (labels table already exists), `src/config.py` (ticker list)
- **Reference:** look at how `price_collector.py` inserts into the DB and logs to `run_log`

**Cheri** — News-to-date linking
- Extend `src/data_processing/standardization.py` to handle weekends and holidays in the cutoff rule
- Currently `apply_cutoff_rule()` maps news → prediction date but doesn't skip weekends
- If news arrives Saturday, it should predict Monday (not Sunday)
- Add a helper to find the next trading day given a date
- Write tests for weekend/holiday cases in `tests/unit/test_standardization.py`
- **Integrates with:** `src/data_processing/standardization.py` (existing cutoff logic)

**Raphael** — Dataset splits
- Create `src/data_processing/dataset_split.py`
- Split data chronologically (NOT randomly — this is time series, random would leak future data)
- 70% train / 15% validation / 15% test by date
- Each split gets both price rows and their associated news articles
- Save split info (which dates go where) to `data/processed/`
- Write tests in `tests/unit/test_dataset_split.py`
- **Integrates with:** `labels` table (needs labels to exist first), `prices` and `news` tables
- **Reference:** `src/config.py` PROCESSED_DATA_DIR for where to save outputs

**Moses & Gordon** — News quality improvements
- Improve the relevance filter in `src/data_collection/news_collector.py`
- Current filter is basic (string match on ticker/company name in headline) — analyze what's getting through and what's being dropped
- Look at article distribution: are some tickers getting almost no articles?
- Document findings for the NLP team to use in Week 4
- If time: add content/summary scraping for articles that have a URL but no body text
- **Reference:** `src/utils/api_clients.py` (FinnhubClient), `src/data_processing/news_validation.py` (distribution check)

### End-of-week checkpoint
- [ ] `labels` table populated for all tickers with collected price data
- [ ] Cutoff rule handles weekends correctly
- [ ] Chronological train/val/test split created and saved
- [ ] All existing tests still pass + new tests added
- [ ] PR merged to main

---

## Week 4 — Feature Engineering

**Goal:** Transform raw data into model-ready inputs.

### Financial team (Raphael + Tim)
- Create `src/features/technical_indicators.py`
  - RSI (Relative Strength Index) — momentum indicator
  - MACD (Moving Average Convergence Divergence) — trend indicator
  - Bollinger Bands — volatility indicator
  - Volume moving averages
- Create `src/features/sequence_generator.py`
  - Build 60-day sliding windows of price + indicator data
  - Each window = one training sample for the LSTM
  - Output shape: (num_samples, 60, num_features) — ready for PyTorch/TensorFlow
- **Integrates with:** `prices` table, `labels` table, `src/config.py` (LSTM_CONFIG has sequence_length=60)
- **Reference:** config.py LSTM_CONFIG for hyperparameters

### NLP team (Moses + Gordon)
- Create `src/features/text_features.py`
  - TF-IDF vectorization of headlines (up to 5000 features, set in NLP_CONFIG)
  - Group articles by (ticker, prediction_date) — one feature vector per prediction day
  - Handle days with no articles (zero vector or flag)
- Optionally start looking at sentence-transformer embeddings for Week 6
- **Integrates with:** `news` table, cutoff rule from `standardization.py`, `src/config.py` (NLP_CONFIG)

### Cheri
- Integration tests: verify feature outputs align with labels (same dates, same tickers)
- Create `scripts/build_features.py` that runs the full pipeline: labels → indicators → sequences → text features

### End-of-week checkpoint
- [ ] Technical indicators computed and spot-checked
- [ ] 60-day sequences generated, shapes verified
- [ ] TF-IDF matrix built from headlines
- [ ] Pipeline script works end-to-end

---

## Week 5 — Baseline Models

**Goal:** First working models. They don't need to be good yet, just functional.

### Financial team (Raphael + Tim)
- Create `src/models/lstm_model.py`
  - 2-layer LSTM (50 units each, 0.2 dropout) — from LSTM_CONFIG
  - Input: 60-day sequences from Week 4
  - Output: probability of next-day up move
  - Training loop with loss tracking
- Create `scripts/train_lstm.py`
- Train on the training split, evaluate on validation split
- Target: anything above 50% directional accuracy means it's learning something

### NLP team (Moses + Gordon)
- Create `src/models/nlp_baseline.py`
  - Logistic Regression on TF-IDF features (scikit-learn)
  - Simple and fast — establishes a baseline
- Create `scripts/train_nlp.py`
- **Integrates with:** text features from Week 4, labels from Week 3

### End-of-week checkpoint
- [ ] LSTM trains without crashing, loss decreases over epochs
- [ ] NLP baseline trained, accuracy computed on validation set
- [ ] Both models save weights to `data/models/`
- [ ] Results documented (even if bad)

---

## Weeks 6–7 — Advanced Models & Evaluation

**Goal:** Improve both models and build proper evaluation tools.

### Financial team
- Hyperparameter tuning: try different sequence lengths, learning rates, LSTM architectures
- Experiment with GRU as an alternative to LSTM
- Use validation set to pick best configuration — do NOT touch test set yet

### NLP team
- Create `src/models/nlp_advanced.py`
  - Replace TF-IDF with FinBERT or sentence-transformer embeddings
  - These capture semantic meaning ("FDA rejects" vs "FDA approves") that TF-IDF misses
- Compare baseline vs advanced on validation set

### Evaluation (Cheri + both teams)
- Create `src/evaluation/metrics.py`
  - Directional accuracy (% of correct up/down predictions)
  - Precision, recall, F1 for each class
  - Confusion matrix
  - Profit simulation (optional): if you traded based on predictions, what's the return?
- Create `src/evaluation/backtester.py`
  - Walk-forward evaluation on the test set
  - Compare model predictions vs actual outcomes day by day
- Store prediction results in the `predictions` table

### End-of-week checkpoint
- [ ] Best LSTM configuration identified
- [ ] Advanced NLP model trained and compared to baseline
- [ ] Evaluation metrics computed for all models
- [ ] Predictions stored in database

---

## Weeks 8–9 — Ensemble Integration

**Goal:** Combine both models into one prediction.

### All team members
- Create `src/models/ensemble.py`
  - Weighted average of LSTM probability + NLP probability
  - Try different weight ratios (e.g., 60/40, 50/50, 70/30)
  - Pick weights that maximize validation accuracy
- Backtest the ensemble on the test set — this is the final accuracy number
- Compare: LSTM alone vs NLP alone vs ensemble
- **Integrates with:** both model outputs, `predictions` table, evaluation metrics

### End-of-week checkpoint
- [ ] Ensemble beats both individual models (ideally)
- [ ] Full backtest results on test set
- [ ] Results table: model / accuracy / precision / recall

---

## Week 10 — Polish

**Goal:** Clean up, fix bugs, boost confidence in results.

- Increase test coverage (aim for all major modules)
- Refactor any messy code from rapid iteration
- Final model training on the full train+validation set, evaluated on test set only
- Write up findings for the final report
- Update this ROADMAP and PROJECT_OVERVIEW with final results

---

## Week 11 — Web Demo

**Goal:** Show it off.

- Create `app/` directory with a Flask web app
  - Search bar: type a ticker symbol
  - Display: prediction (up/down), confidence %, model breakdown (LSTM vs NLP)
  - Show top 3-5 headlines that influenced the NLP prediction
- Deploy locally or on a simple hosting service
- Record a demo video or prepare a live presentation

### End-of-week checkpoint
- [ ] App runs locally
- [ ] All 15 tickers produce predictions
- [ ] Looks presentable for a demo

---

## Data Standards (Quick Reference)

These are enforced in the code (`schema.py`, `standardization.py`, collectors):

| Standard              | Format                            | Where enforced           |
|-----------------------|-----------------------------------|--------------------------|
| Ticker symbols        | Uppercase (`AAPL`)                | config.py, collectors    |
| Price dates           | `YYYY-MM-DD`                      | standardization.py       |
| News timestamps       | ISO-8601 ET (`2026-02-07T14:13:00-05:00`) | standardization.py |
| Missing values        | `NULL` (not empty string)         | schema.py                |
| News deduplication    | URL is unique key                 | schema.py                |
| Price deduplication   | (ticker, date) is unique          | schema.py                |
| News cutoff           | Before 4PM ET → predicts T+1, after → predicts T+2 | standardization.py |
