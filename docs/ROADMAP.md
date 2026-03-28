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
├── config.py                       # settings, ticker list, API key
├── database/schema.py              # 5 tables: prices, news, labels, predictions, run_log
├── data_collection/
│   ├── price_collector.py          # Yahoo Finance → prices table
│   └── news_collector.py          # Finnhub API → news table (word-boundary relevance filter)
├── data_processing/
│   ├── standardization.py          # date/time utils, cutoff rule (skips weekends/holidays)
│   ├── price_validation.py         # quality checks on prices
│   ├── news_validation.py          # quality checks on news
│   ├── label_generator.py          # up/down labels from price data
│   └── dataset_split.py            # chronological train/val/test split
├── features/
│   ├── technical_indicators.py     # RSI, MACD, Bollinger Bands, volume MA
│   ├── sequence_generator.py       # 60-day sliding windows for LSTM input
│   └── text_features.py            # TF-IDF extraction from news headlines
├── models/
│   ├── lstm_model.py               # 2-layer LSTM + trainer (PyTorch)
│   └── nlp_baseline.py             # logistic regression on TF-IDF (scikit-learn)
└── evaluation/                     # (Week 6+) accuracy metrics, backtesting

scripts/                            # CLI entry points (python scripts/xxx.py)
  train_lstm.py                     # train LSTM on price sequences
  train_nlp.py                      # train NLP baseline on headlines
tests/unit/                         # pytest tests
data/                               # git-ignored — database, processed features, model weights
docs/                               # this file, project overview
```

---

## Week 1 — Setup ✓

Config, database schema, project structure, requirements.

## Week 2 — Data Collection ✓

Price + news collectors, validation, standardization, demo script.

## Week 3 — Labels & Splits ✓

| Who     | Task | Status |
|---------|------|--------|
| Tim     | Generate up/down labels from price data | ✓ `label_generator.py` |
| Raphael | Chronological train/val/test split (70/15/15) | ✓ `dataset_split.py` |
| Cheri   | Weekend/holiday handling in cutoff rule | ✓ `standardization.py` |
| Moses + Gordon | Improve news relevance filter | ✓ `news_collector.py` |

## Week 4 — Features ✓

| Who     | Task | Status |
|---------|------|--------|
| Raphael + Tim | Technical indicators (RSI, MACD, Bollinger) + 60-day LSTM sequences | ✓ `technical_indicators.py`, `sequence_generator.py` |
| Moses + Gordon | TF-IDF on headlines | ✓ `src/features/text_features.py` |
| Cheri   | Pipeline script (+ sequence date metadata) | ✓ `scripts/build_features.py` |

**Note:** sequences need 60+ days of price data. Collect more with `python scripts/collect_prices.py --days 90`.

## Week 5 — Baseline Models ✓

| Who     | Task | Status |
|---------|------|--------|
| Raphael + Tim | 2-layer LSTM on price sequences | ✓ `src/models/lstm_model.py`, `scripts/train_lstm.py` |
| Moses + Gordon | Logistic regression on TF-IDF | ✓ `src/models/nlp_baseline.py`, `scripts/train_nlp.py` |

## Weeks 6–7 — Tuning & Evaluation

| Who     | Task | File to create |
|---------|------|----------------|
| Raphael + Tim | Hyperparameter tuning (LSTM epochs, lr, units, dropout) | update `scripts/train_lstm.py` |
| Raphael + Tim | Evaluation metrics (accuracy, precision, recall, F1, confusion matrix) | `src/evaluation/metrics.py` |
| Moses + Gordon | FinBERT / sentence-transformer embeddings | `src/models/nlp_advanced.py` |
| Moses + Gordon | Compare NLP baseline vs advanced on same test set | update `scripts/train_nlp.py` |
| Cheri | Backtesting framework (simulated trades from predictions) | `src/evaluation/backtester.py` |

**Prerequisite:** collect enough data first — `python scripts/demo.py --all --days 365`.

## Weeks 8–9 — Ensemble

Combine LSTM + NLP predictions, optimize weights, backtest. File: `src/models/ensemble.py`.

## Week 10 — Polish

Bug fixes, test coverage, final training run, report writeup.

## Week 11 — Web Demo

Flask app: search a ticker, see prediction + confidence + supporting headlines. Directory: `app/`.
