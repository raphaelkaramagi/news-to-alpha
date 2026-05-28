# Data & predictions

How data flows through the Stock Price and Sentiment Predictor pipeline, what each artifact does, and how to keep predictions current.

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
| `evaluation_*.csv` | Accuracy metrics (`all`, `has_news`, `high_conf` subsets) |
| `pipeline_config.json` | Last train config (horizon, seeds, tickers) |
| `last_published.json` | Timestamp written by publish script |

### `data/models/`

| File | Purpose |
|------|---------|
| `lstm_model.pt` | LSTM weights + scaler (+ optional seed models) |
| `nlp_baseline.joblib` | TF-IDF + logistic |
| `news_embeddings.joblib` | FinBERT (max_v2) or MiniLM + classifier |
| `ensemble_meta.joblib` | Meta-model(s) + feature importances (Why tab) |

---

## Why predictions lag prices

Labels need the **next trading day's close**. After Memorial Day 2025-style gaps, the last *labeled* date can be earlier than the last *price* date.

**Live inference** scores dates after the last label using saved model weights:

| Module | What it does |
|--------|----------------|
| `src/ml/lstm_live_export.py` | LSTM rows for unlabeled price dates |
| `src/ml/news_live_export.py` | TF-IDF + FinBERT for live dates with headlines |
| `score_models.py` → `backfill_outcomes()` | Fills `actual_binary` on live rows once labels exist |

After each close, the pipeline also emits a **forward session** forecast (e.g. on May 28 morning you see a May 28 forecast using data through May 27 close).

### When outcomes resolve (important)

A forecast for **date T** predicts the move from **close(T) → close(T+1)**.

| You are viewing | Needs in DB | UI shows |
|-----------------|-------------|----------|
| May 26 | May 27 close | ✓ / ✗ once labels + backfill run |
| May 27 | May 28 close | Pending until May 28 close + `daily_update` |
| May 28 (today) | May 28 close not yet | Forward forecast; pending until May 29 close |

`has_news` is derived from **cutoff-aligned headline count** in SQLite (`n_headlines > 0`), not from whether a news-model CSV row existed at train time.

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

### First-time or full retrain (production)

Use the **`max`** preset before your first deploy — all 20 tickers, 3 years of prices, next-day horizon (matches the UI), FinBERT embeddings, 3-seed LSTM with 120 epochs:

```bash
source .venv/bin/activate
cp .env.example .env   # NEWS_API_KEY=...

python scripts/setup_database.py
python scripts/run_pipeline.py --preset max
```

Retrain **replaces** prediction CSVs (no duplicate rows). The pipeline also prunes stale `predictions` DB rows and removes old `lstm_model_seed*.pt` files so seed ensembles stay in sync.

Other presets:

| Preset | Use case |
|--------|----------|
| `quick` | 2 tickers, smoke test (~minutes) |
| `balanced` | 5 tickers, 3-day horizon, faster iteration |
| `advanced` | All tickers, 3-day horizon, FinBERT |
| **`max`** | **All tickers, next-day — current production deploy preset** |
| **`max_v2`** | **Accuracy experiment: FinBERT, conditional ensemble, 0.3% min-move filter, VIX features, 150 LSTM epochs** |

`max_v2` LSTM changes (May 2026): weighted BCE (not focal loss), val **AUC** early-stop, sigmoid calibration when raw proba variance is low. Retrain LSTM only:

```bash
python scripts/run_pipeline.py --preset max_v2 \
  --skip-collect --skip-news --skip-nlp --skip-emb --skip-ensemble --skip-evaluate
python scripts/build_eval_dataset.py
python scripts/build_ensemble.py --conditional
python scripts/evaluate_predictions.py --horizon 1
```

```bash
# Full retrain with all accuracy improvements (skips price re-download if already current)
python scripts/run_pipeline.py --preset max_v2 --skip-collect --skip-news

# Backfill historical news (Finnhub free tier ~1 year; always chunk + fill gaps):
python scripts/collect_news.py --days 365 --backfill --fill-gaps --chunk-days 7

# Repair a sparse month (Finnhub caps ~240 articles per call without chunking):
python scripts/collect_news.py --start-date 2026-05-01 --end-date 2026-05-27 --fill-gaps
```

This runs: collect → labels → split → train all models → ensemble → evaluate, and appends **live** LSTM rows automatically at the end of `train_lstm.py`.

### Daily refresh (no retrain)

```bash
python scripts/daily_update.py
```

Steps: collect prices/news → labels → `score_models.py` (live LSTM + live news + outcome backfill) → rebuild ensemble → evaluate.

Dry-run first:

```bash
python scripts/daily_update.py --dry-run
```

### Mac auto-update (launchd)

`scripts/local_cron.sh` runs `daily_update.py` then publishes to Railway. Schedule: **Mon–Fri 22:00 UTC** (~6 PM ET in EDT).

**One-time setup (macOS Ventura+):**

```bash
cp docs/local_cron.plist.example ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
# Edit WorkingDirectory + script path if needed
plutil -lint ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
launchctl bootout gui/$(id -u)/com.news-to-alpha.daily 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
launchctl enable gui/$(id -u)/com.news-to-alpha.daily
launchctl print gui/$(id -u)/com.news-to-alpha.daily
```

Logs: `tail -f ~/Library/Logs/news-to-alpha/daily_update.log`

**Requirements:** Mac awake at scheduled time, `.venv` + `.env` in project root, `railway` CLI logged in. Missed runs: `bash scripts/local_cron.sh`.

Dry-run:

```bash
bash scripts/local_cron.sh --dry-run
```

### Quick catch-up (collect + live score only)

If you already have trained models and only need to advance dates:

```bash
python scripts/collect_prices.py --days 14
python scripts/collect_news.py --days 14 --fill-gaps
python scripts/generate_labels.py
python scripts/score_models.py          # live LSTM + live news + backfill outcomes
python scripts/build_eval_dataset.py
python scripts/build_ensemble.py --conditional
python scripts/evaluate_predictions.py --horizon 1
```

Verify:

```bash
python scripts/audit_data_coverage.py
python -c "import pandas as pd; df=pd.read_csv('data/processed/final_ensemble_predictions.csv'); print('max', df['prediction_date'].max()); print('has_news mismatches', ((df['has_news']==0)&(df['n_headlines']>0)).sum())"
curl -s http://127.0.0.1:8000/api/data-status | python3 -m json.tool
```

Expected: `latest_prediction_date` ≥ latest price date; forward session (e.g. today) may appear before today's close. `latest_resolved_prediction_date` lags by one session until T+1 price exists.

### Publish to Railway

Volume on the **web** service must be mounted at **`/data`**. Then:

```bash
python scripts/publish_deploy_bundle.py --dry-run
python scripts/publish_deploy_bundle.py --target railway --service web
```

Requires `railway login`, `railway link` (service: **web**), registered SSH key (`railway ssh keys add`), and the service **Online**. Upload uses `railway ssh` (not `railway run` — run is local-only and has no volume). If `Host key verification failed`, run `ssh-keyscan ssh.railway.com >> ~/.ssh/known_hosts`. If writes fail, add Railway variable `RAILWAY_RUN_UID=0`.

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
| `latest_prediction_date` | Newest row in ensemble CSV (includes forward session) |
| `latest_resolved_prediction_date` | Newest date where most tickers have `actual_binary` filled |
| `latest_price_date` | Newest price in SQLite |
| `expected_latest_prediction_date` | Same as latest price — where preds should reach |
| `trading_sessions_behind` | Sessions between pred max and price max |
| `is_current` | `true` when predictions cover through latest price |
| `market_status` | `open` / `closed` / `pre_market` (4 PM ET cutoff) |
| `pending_reason` | `awaiting_next_close` / `awaiting_data_refresh` / `resolved` |

UI: Markets banner uses `pending_reason` per selected date. Freshness badge shows forecasts vs resolved-through dates.

---

## News collection (Finnhub)

Finnhub returns **at most ~240 articles per API call**. A single wide date range keeps only the newest articles — older days in the month are dropped silently.

| Mode | Command |
|------|---------|
| Daily / pipeline | `collect_news.py --days N --fill-gaps` (7-day chunks + per-day gap fill) |
| Backfill history | `--days 365 --backfill --fill-gaps --chunk-days 7` |
| Repair one month | `--start-date YYYY-MM-01 --end-date YYYY-MM-DD --fill-gaps` |

Always use `--fill-gaps` for production collection. Without it, you can get clusters on recent days (e.g. 36 headlines on May 26, zero on May 4–20) even when Finnhub has data for those days.

---

| Script | When to use |
|--------|-------------|
| `run_pipeline.py` | Full retrain (`--preset max` or `max_v2`) |
| `daily_update.py` | Weekday refresh without retrain |
| `score_models.py` | Live LSTM + news scoring + outcome backfill |
| `audit_data_coverage.py` | Price/news/label/live-row consistency report |
| `publish_deploy_bundle.py` | Trim + upload to Railway |
| `collect_prices.py` / `collect_news.py` | Manual data pull |
| `generate_labels.py` | Rebuild labels after new prices |

See also: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) (architecture deep dive).

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

**Layperson view (`Why this call`):** direction, confidence, combiner route (news-tuned vs price-only), plain summary, and three input pills. On days without headlines, Keywords and FinBERT show “No headlines” instead of a fake 50% score.

**Advanced tab:** per-model probabilities, all 13 ensemble inputs, LSTM context snapshot, counterfactual driver bars, rolling accuracy.

The ensemble uses counterfactual explanation (`src/ml/ensemble_explain.py`). With `--conditional`, rows with headlines use a separate meta-model; the Advanced tab notes which route applied.

Restart Flask after retraining so driver bars load (client fallback shows votes only):

```bash
python app/server.py --port 8000
```

### Evaluation subsets

`evaluate_predictions.py` reports three test slices:

| Subset | Meaning |
|--------|---------|
| `all` | Every test row |
| `has_news` | Days with at least one headline |
| `high_conf` | Ensemble confidence ≥ 0.25 |

Check `/status` in the UI or `data/processed/evaluation_summary.txt`.
