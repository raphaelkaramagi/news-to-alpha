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
│   └── nlp_baseline.py             # logistic regression on TF-IDF (scikit-learn)
└── evaluation/                     # [Week 10] metrics, evaluation scripts

scripts/
├── train_lstm.py                   # LSTM train + export → price_predictions.csv
├── train_nlp.py                    # TF-IDF baseline train + export → news_tfidf_predictions.csv
├── train_news_embeddings.py        # Sentence embeddings + LR → news_embeddings_predictions.csv
├── demo.py                         # end-to-end pipeline
└── ...                             # data collection, labels, features, reset

tests/unit/                         # 83 automated tests (pytest)
data/                               # git-ignored — database, features, model weights
docs/                               # roadmap, project overview, contracts
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

## Weeks 6–8 — Evaluation, Improved Models & Contracts ✓

| Who            | Task | Status |
|----------------|------|--------|
| Raphael + Tim  | LSTM hyperparameter tuning (gradient clipping, LR scheduling, weight init) | ✓ Done |
| Raphael + Tim  | Per-class recall and accuracy reporting in training scripts | ✓ Done |
| Moses          | Rewrite `train_nlp.py` — cutoff-aligned pipeline with CSV export per session 1 contract | ✓ Done |
| Gordon         | Sentence-embedding NLP model (`train_news_embeddings.py`) — MiniLM + logistic regression | ✓ Done |
| Cheri          | Session 1 contract (`docs/session1_contract.md`) — shared output schema, join key, cutoff rule | ✓ Done |
| All            | 83 unit tests covering all modules | ✓ Done |

## Week 9 — Model Training & Prediction Export ✓

**Goal:** Every model produces a standardized prediction CSV keyed by `(ticker, prediction_date)`, ready for ensemble integration.

| Who            | Task | Status |
|----------------|------|--------|
| Raphael        | Extend `train_lstm.py` — train + evaluate + export to `price_predictions.csv` with `financial_pred_proba`, `financial_confidence`, `model_name = "lstm_price"` | ✓ Done |
| Tim            | `check_price_alignment.py` — QC script validating sequence/split/label consistency | ✓ Done |
| Moses          | `train_nlp.py` — TF-IDF baseline export to `news_tfidf_predictions.csv` | ✓ Done |
| Gordon         | `train_news_embeddings.py` — embeddings export to `news_embeddings_predictions.csv` | ✓ Done |

**Artifacts produced by Week 9:**

| File | Owner |
|------|-------|
| `data/models/lstm_model.pt` | Raphael |
| `data/processed/price_predictions.csv` | Raphael |
| `data/models/nlp_baseline.joblib` | Moses |
| `data/processed/news_tfidf_predictions.csv` | Moses |
| `data/models/news_embeddings.joblib` | Gordon |
| `data/processed/news_embeddings_predictions.csv` | Gordon |
| `data/processed/price_alignment_report.txt` | Tim |

---

## Week 10 — Ensemble Integration, Evaluation & App

**Goal:** Join the three model outputs into one aligned dataset, compute evaluation metrics, build the ensemble, and wire up the demo app.

| Who            | Task |
|----------------|------|
| Raphael        | `scripts/build_eval_dataset.py` — inner-join all three prediction CSVs on `(ticker, prediction_date)`, create `eval_dataset.csv` |
| Tim            | `scripts/evaluate_predictions.py` — accuracy, precision, recall, F1 for each model + ensemble + baselines; save overall and per-ticker reports |
| Moses          | `scripts/build_ensemble.py` — combine model probabilities using the locked formula from `session_2_contract.md`; save `final_ensemble_predictions.csv` |
| Gordon         | `app.py` — Streamlit app connected to `final_ensemble_predictions.csv` (ticker dropdown, prediction, confidence, headlines) |
| Cheri          | `docs/session_2_contract.md` — lock ensemble formula, output file list, demo tickers, app priority order |

---

## Week 11 — Presentation Prep & Polish

Google Slides, deciding who presents what, app polish, fixing demo-breaking bugs only, rehearsal.
