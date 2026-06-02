# Project overview

Architecture for contributors. Commands: [DATA.md](DATA.md). Metrics: [RESULTS.md](RESULTS.md).

---

## Models

| Model | Input | Output |
|-------|--------|--------|
| **LSTM** | 60-day window, technicals, ticker embedding, SPY/VIX | P(UP) — test AUC ≈ 0.50 |
| **TF-IDF** | Cutoff-aligned headlines + publisher one-hot | P(UP) |
| **FinBERT embeddings** | Mean-pooled headline vectors | P(UP) |
| **Volatility** | Realized vol, ATR, BB width, gaps, VIX, etc. | `expected_move_pct` (|return|) — high-move AUC ≈ 0.65 |
| **Ensemble** | 13 meta features, conditional has_news / no_news | Final P(UP) — test AUC ≈ 0.50 |

Direction is the headline product; expected move is the complementary signal.

---

## Pipeline

```mermaid
flowchart TD
  DB[(SQLite prices + news)] --> LSTM[train_lstm]
  DB --> NLP[train_nlp + train_news_embeddings]
  DB --> VOL[train_volatility]
  LSTM --> P1[price_predictions.csv]
  NLP --> P2[news_*_predictions.csv]
  VOL --> P3[volatility_predictions.csv]
  P1 --> ED[build_eval_dataset]
  P2 --> ED
  P3 --> ED
  ED --> BE[build_ensemble --conditional]
  BE --> OUT[final_ensemble_predictions.csv]
  score[score_models.py daily] --> P1
  score --> P2
  score --> P3
  OUT --> API[Flask]
  API --> UI[Next.js]
```

**Live scoring:** `run_pipeline.py` and `score_models.py` append `split=live` rows for LSTM, news, and volatility after each session.

---

## Evaluation

- Chronological 70/15/15 split (LSTM anchor).
- News scores joined only when news model split matches row split.
- Skill metric: **AUC** on direction; MAE + high-move AUC for volatility.
- Production: `INFERENCE_ONLY=true` on Railway — no web-triggered training.

---

## API surface

Key routes: `/api/ticker` (direction + expected move band), `/api/rationale`, `/api/data-status`, `/api/history`.

Payload additions: `expected_move_pct`, `forecast_low`, `forecast_high`, `actual_abs_return_pct`.
