# Roadmap

## Team

| Person  | Focus                    | Reviews with |
|---------|--------------------------|--------------|
| Raphael | Prices, LSTM             | Tim          |
| Tim     | Database, LSTM training  | Raphael      |
| Moses   | News, NLP baseline/advanced | Gordon    |
| Gordon  | API clients, embeddings  | Moses        |
| Cheri   | Integration, ensemble    | cross-team   |

---

## Project Structure

```
src/
├── config.py                   # settings, ticker list, API key
├── database/schema.py          # 5 tables: prices, news, labels, predictions, run_log
├── data_collection/
│   ├── price_collector.py      # Yahoo Finance → prices table
│   └── news_collector.py       # Finnhub API → news table
├── data_processing/
│   ├── standardization.py      # date/time utils, cutoff rule
│   ├── price_validation.py     # quality checks on prices
│   ├── news_validation.py      # quality checks on news
│   ├── label_generator.py      # (Week 3 - Tim) up/down labels from price data
│   └── dataset_split.py        # (Week 3 - Raphael) chronological train/val/test split
├── features/                   # (Week 4) technical indicators, LSTM sequences, TF-IDF
├── models/                     # (Week 5+) LSTM, NLP baseline, NLP advanced, ensemble
└── evaluation/                 # (Week 6+) accuracy metrics, backtesting

scripts/                        # CLI entry points (python scripts/xxx.py)
tests/unit/                     # pytest tests
data/                           # git-ignored — database, raw files, model weights
docs/                           # this file, project overview, git guide
```

Empty folders (`features/`, `models/`, `evaluation/`) have `__init__.py` files so Python treats them as packages. Code goes there in later weeks.

---

## Week 1 — Setup ✓

Config, database schema, project structure, requirements.

## Week 2 — Data Collection ✓

Price + news collectors, validation, standardization, demo script, 13 tests.

## Week 3 — Labels & Splits (current)

| Who     | Task | File to create |
|---------|------|----------------|
| Tim     | Generate up/down labels from price data | `src/data_processing/label_generator.py`, `scripts/generate_labels.py` |
| Raphael | Chronological train/val/test split (70/15/15) | `src/data_processing/dataset_split.py` ✓, `scripts/split_dataset.py` ✓ |
| Cheri   | Weekend/holiday handling in cutoff rule | update `src/data_processing/standardization.py` |
| Moses + Gordon | Improve news relevance filter | update `src/data_collection/news_collector.py` |

## Week 4 — Features

| Who     | Task | File to create |
|---------|------|----------------|
| Raphael + Tim | Technical indicators (RSI, MACD, Bollinger), 60-day sequences for LSTM | `src/features/technical_indicators.py`, `src/features/sequence_generator.py` |
| Moses + Gordon | TF-IDF on headlines, group by (ticker, date) | `src/features/text_features.py` |
| Cheri   | Pipeline script tying it all together | `scripts/build_features.py` |

## Week 5 — Baseline Models

| Who     | Task | File to create |
|---------|------|----------------|
| Raphael + Tim | 2-layer LSTM on price sequences | `src/models/lstm_model.py`, `scripts/train_lstm.py` |
| Moses + Gordon | Logistic regression on TF-IDF | `src/models/nlp_baseline.py`, `scripts/train_nlp.py` |

## Weeks 6–7 — Tuning & Evaluation

Hyperparameter tuning (LSTM), FinBERT/embeddings (NLP), evaluation framework with accuracy/precision/recall/confusion matrix. Files: `src/models/nlp_advanced.py`, `src/evaluation/metrics.py`, `src/evaluation/backtester.py`.

## Weeks 8–9 — Ensemble

Combine LSTM + NLP predictions, optimize weights, backtest. File: `src/models/ensemble.py`.

## Week 10 — Polish

Bug fixes, test coverage, final training run, report writeup.

## Week 11 — Web Demo

Flask app: search a ticker → see prediction + confidence + supporting headlines. Directory: `app/`.
