# Data & predictions

How data flows through News-to-Alpha, what each artifact does, and how to keep predictions current.

## Architecture (local train → lean serve)

| Layer | Location | Purpose |
|-------|----------|---------|
| **Collect + train** | Your Mac | Full pipeline, heavy PyTorch training |
| **Daily refresh** | Your Mac (cron) | Collect prices/news, score live dates, publish bundle |
| **API** | Railway (`Dockerfile.inference`) | Read CSVs + SQLite, optional infer fallback |
| **UI** | Vercel (`web/`) | Next.js read-only dashboard |

Training data and raw downloads stay on your machine. Only inference artifacts go to Railway.

---

## Key directories & files

### `data/database.db` (SQLite)

| Table | Contents |
|-------|----------|
| `prices` | Daily OHLCV per ticker |
| `news` | Headlines, source, URL, published_at |
| `labels` | Up/down labels + `%` return (needs next trading close) |

Used by: collection scripts, headlines API, chart price line, resolved-call badges.

### `data/processed/` (CSVs)

| File | Purpose |
|------|---------|
| `price_predictions.csv` | LSTM per (ticker, date) |
| `news_tfidf_predictions.csv` | TF-IDF news model scores |
| `news_embeddings_predictions.csv` | Embedding news scores |
| `eval_dataset.csv` | Joined per-model CSVs |
| **`final_ensemble_predictions.csv`** | **What the UI reads** — ensemble + all model columns |
| `evaluation_*.csv` | Accuracy / conviction metrics |
| `pipeline_config.json` | Last train config (horizon, seeds, tickers) |
| `last_published.json` | Timestamp written by publish script |

### `data/models/`

| File | Purpose |
|------|---------|
| `lstm_model.pt` | LSTM weights + scaler (+ optional seed models) |
| `nlp_baseline.joblib` | TF-IDF + logistic |
| `news_embeddings.joblib` | MiniLM + classifier |
| `ensemble_meta.joblib` | Meta-model + feature importances (Why tab) |

---

## Why predictions lag prices

Labels need the **next trading day's close**. After Memorial Day 2025-style gaps, the last *labeled* date can be earlier than the last *price* date.

**Live inference** (`src/ml/lstm_live_export.py`) scores dates after the last label using saved LSTM weights — e.g. prices through **2026-05-26** get predictions even before that day's move is known.

### Why dates skip (e.g. 5/22 → 5/26)

The date scrubber only lists **NYSE trading sessions** that have a prediction row — not every calendar day.

Example around Memorial Day 2026:

| Calendar date | Session? | In scrubber? |
|---------------|----------|--------------|
| Fri 5/22 | Yes | Yes |
| Sat 5/23 | Weekend | No |
| Sun 5/24 | Weekend | No |
| Mon 5/25 | Memorial Day (closed) | No |
| Tue 5/26 | Yes | Yes |

So jumping from 5/22 to 5/26 is expected — there were no trading sessions in between.

---

## Updating data locally

### First-time or full retrain

```bash
source .venv/bin/activate
cp .env.example .env   # NEWS_API_KEY=...

python scripts/setup_database.py
python scripts/run_pipeline.py --preset balanced
```

This runs: collect → labels → split → train all models → ensemble → evaluate, and appends **live** LSTM rows automatically at the end of `train_lstm.py`.

### Daily refresh (no retrain)

```bash
python scripts/daily_update.py
```

Steps: collect prices/news → labels → `score_models.py` (live LSTM) → rebuild ensemble → evaluate.

Dry-run first:

```bash
python scripts/daily_update.py --dry-run
```

### Quick catch-up (collect + live score only)

If you already have trained models and only need to advance dates:

```bash
python scripts/collect_prices.py --days 14
python scripts/collect_news.py --days 14
python scripts/generate_labels.py
python scripts/score_models.py          # appends live LSTM rows
python scripts/build_eval_dataset.py
python scripts/build_ensemble.py
python scripts/evaluate_predictions.py --horizon 1
```

Verify:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/processed/final_ensemble_predictions.csv')['prediction_date'].max())"
curl -s http://127.0.0.1:8000/api/data-status | python3 -m json.tool
```

Expected: `latest_prediction_date` matches `latest_price_date` (e.g. `2026-05-26`).

### Publish to Railway

```bash
python scripts/publish_deploy_bundle.py --dry-run
python scripts/publish_deploy_bundle.py --target railway
```

Or use the Mac cron wrapper: `scripts/local_cron.sh` (see `docs/local_cron.plist.example` — local only, gitignored).

---

## Running locally for development

**Terminal 1 — API**

```bash
source .venv/bin/activate
python app/server.py --port 8000
```

**Terminal 2 — UI**

```bash
cd web
cp .env.example .env.local   # API_BASE_URL=http://127.0.0.1:8000
npm install && npm run dev
```

Open http://localhost:3000 — Markets, `/t/AAPL`, `/status`.

---

## Freshness fields (`/api/data-status`)

| Field | Meaning |
|-------|---------|
| `latest_prediction_date` | Newest row in ensemble CSV |
| `latest_price_date` | Newest price in SQLite |
| `expected_latest_prediction_date` | Same as latest price — where preds should reach |
| `trading_sessions_behind` | Sessions between pred max and price max |
| `is_current` | `true` when predictions cover through latest price |

---

## Scripts reference

| Script | When to use |
|--------|-------------|
| `run_pipeline.py` | Full retrain |
| `daily_update.py` | Weekday refresh without retrain |
| `score_models.py` | Live LSTM scoring only |
| `publish_deploy_bundle.py` | Trim + upload to Railway |
| `collect_prices.py` / `collect_news.py` | Manual data pull |
| `generate_labels.py` | Rebuild labels after new prices |

See also: [DEPLOY.md](DEPLOY.md) (Railway API), [DEPLOY_UI.md](DEPLOY_UI.md) (Vercel + full stack).

---

## “Why this call” & confidence

### Confidence score

Every model uses the same formula:

```text
confidence = |probability − 0.5| × 2    # range 0–1
```

- **0%** = exactly 50/50 (coin flip)
- **100%** = 0% or 100% UP (extreme lean)

This measures **how strong the lean is**, not the chance the call is correct. A **DOWN** call at 27% UP can still show **~46% confidence** because 27% is far from 50%.

Labels in the UI: **Low** (&lt;25%), **Moderate** (25–45%), **Strong** (≥45%).

### Why this call tab

The ensemble meta-model is **HistGradientBoosting** (non-linear). The UI uses counterfactual explanation (`src/ml/ensemble_explain.py`):

1. Final call + confidence (one card)
2. Short summary (1–2 lines) — model split + simple avg vs ensemble
3. Three vote pills — Price / Keywords / Meaning raw scores
4. **What moved the final score** — counterfactual bars per factor

**Headline rows show “No impact”** when neutralizing that feature barely changes the ensemble (common when validation permutation importance for news features is ~0). News models still run and appear in the vote pills; the combiner simply did not rely on them for this call.

Client-side fallback (`web/lib/ensembleExplainClient.ts`) fills in votes when Flask is stale; restart Flask for full driver bars:

```bash
python app/server.py --port 8000
```
