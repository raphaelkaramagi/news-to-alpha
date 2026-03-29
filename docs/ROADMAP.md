# Roadmap

## Team

| Person  | Focus                          | Reviews with |
|---------|--------------------------------|--------------|
| Raphael | Prices, LSTM                   | Tim          |
| Tim     | Database, LSTM training        | Raphael      |
| Moses   | News, NLP (baseline + advanced)| Gordon       |
| Gordon  | API clients, embeddings        | Moses        |
| Cheri   | Integration, ensemble          | cross-team   |

---

## Project Structure

```
src/
├── config.py                       # settings, ticker list, API key, hyperparameters
├── database/schema.py              # 5 tables: prices, news, labels, predictions, run_log
├── data_collection/
│   ├── price_collector.py          # Yahoo Finance → prices table
│   └── news_collector.py          # Finnhub API → news table
├── data_processing/
│   ├── standardization.py          # date/time utils, 4PM ET cutoff rule
│   ├── price_validation.py         # data quality checks on prices
│   ├── news_validation.py          # data quality checks on news
│   ├── label_generator.py          # up/down labels from consecutive closing prices
│   └── dataset_split.py            # chronological train/val/test split (70/15/15)
├── features/
│   ├── technical_indicators.py     # RSI, MACD, Bollinger Bands, volume MA, daily return
│   ├── sequence_generator.py       # 60-day sliding windows → LSTM training samples
│   └── text_features.py            # TF-IDF extraction from news headlines
├── models/
│   ├── lstm_model.py               # 2-layer LSTM + trainer (PyTorch)
│   ├── nlp_baseline.py             # logistic regression on TF-IDF (scikit-learn)
│   ├── nlp_advanced.py             # [Week 6+] FinBERT / sentence embeddings
│   └── ensemble.py                 # [Week 8+] combine LSTM + NLP predictions
└── evaluation/
    ├── metrics.py                  # [Week 6+] accuracy, AUC-ROC, F1, confusion matrix
    └── backtester.py               # [Week 7+] simulated trading returns

scripts/                            # CLI entry points (python scripts/xxx.py --help)
tests/unit/                         # pytest tests
data/                               # git-ignored — database, features, model weights
docs/                               # this file, project overview
```

---

## Week 1 — Setup ✓

Config, database schema, project structure, requirements, git workflow.

## Week 2 — Data Collection ✓

Price + news collectors, data validation, standardization, initial demo script.

## Week 3 — Labels & Splits ✓

| Who            | Task                                          |
|----------------|-----------------------------------------------|
| Tim            | Up/down labels from price data                |
| Raphael        | Chronological train/val/test split (70/15/15) |
| Cheri          | Weekend/holiday handling in the cutoff rule   |
| Moses + Gordon | Improved news relevance filtering             |

## Week 4 — Features ✓

| Who            | Task                                                      |
|----------------|-----------------------------------------------------------|
| Raphael + Tim  | Technical indicators (RSI, MACD, Bollinger) + 60-day LSTM sequences |
| Moses + Gordon | TF-IDF feature extraction from headlines                  |
| Cheri          | `build_features.py` pipeline + sequence date metadata     |

## Week 5 — Baseline Models ✓

| Who            | Task                                       |
|----------------|--------------------------------------------|
| Raphael + Tim  | 2-layer LSTM on price sequences            |
| Moses + Gordon | Logistic regression on TF-IDF headlines    |

**Current results** (all 15 tickers, 250 days of price history):

| Model        | Test accuracy | vs. random |
|--------------|--------------|------------|
| LSTM         | ~51%         | +1%        |
| NLP baseline | ~53%         | +3%        |

Both models show weak but real signal. Expected for a Week 5 baseline using only public data. The ensemble (Week 8) should improve both.

**Note on news data:** Free-tier APIs return only the last ~21 days of articles. The NLP model uses its own independent date split on whatever news is available. Run `python scripts/collect_news.py` weekly to accumulate more training data.

---

## Weeks 6–7 — Evaluation & Improved Models

**Goal:** Understand where each model is right and wrong, improve NLP signal, and set up a backtesting framework to translate accuracy into trading returns.

| Who            | Task |
|----------------|------|
| Raphael + Tim  | Proper evaluation metrics beyond accuracy — think about what signals that the model is actually learning vs. getting lucky. How do you measure this? |
| Raphael + Tim  | Hyperparameter search for LSTM — what knobs to turn, how to search efficiently without overfitting to val set? |
| Moses + Gordon | Replace TF-IDF with a financial language model (FinBERT or similar) — how do you handle the small training set? fine-tuning vs. frozen embeddings? |
| Moses + Gordon | Compare new NLP model vs. baseline on the same test set |
| Cheri          | Backtesting framework — if we traded based on model predictions, what return would we get vs. buy-and-hold? |

**Prerequisite:** Collect enough news data first — `python scripts/collect_news.py` weekly, and run `python scripts/demo.py --days 365` for longer price history.

---

## Weeks 8–9 — Ensemble

**Goal:** Combine LSTM and NLP predictions into a single signal that's stronger than either alone.

Questions to answer: How do you combine two probability scores? Weighted average, learned meta-model, or something else? How do you handle dates where news is missing (LSTM only) vs. dates with news?

File to create: `src/models/ensemble.py`

---

## Week 10 — Polish & Report

Final training run on all data. Improve test coverage. Write up results — what worked, what didn't, what would you do with more time/data?

---

## Week 11 — Web Demo

Flask app: search a ticker → see predicted direction, confidence score, and the headlines that drove the NLP prediction.

Directory: `app/`
