# Data & predictions

How data flows through the pipeline, what each artifact stores, and how to keep predictions current.

## Architecture

| Layer | Location | Purpose |
|-------|----------|---------|
| **Collect + train** | Operator host (local) | Full pipeline, PyTorch training |
| **Daily refresh** | Operator host (cron optional) | Collect prices/news, score live dates, publish bundle |
| **API** | Railway (`Dockerfile.inference`) | Read CSVs + SQLite from `/data` volume |
| **UI** | Vercel (`web/`) | Read-only dashboard |

Raw downloads and training stay on the operator machine. Only inference artifacts are uploaded to the API host.

---

## Key directories & files

### `data/database.db` (SQLite)

| Table | Contents |
|-------|----------|
| `prices` | Daily OHLCV per ticker |
| `news` | Headlines, source, URL, published_at |
| `labels` | Up/down labels + `%` return (requires next session close) |
| `predictions` | Optional mirror of model CSV rows |
| `run_log` | Script execution audit trail |

### `data/processed/` (CSVs)

| File | Purpose |
|------|---------|
| `price_predictions.csv` | LSTM per (ticker, date) |
| `news_tfidf_predictions.csv` | TF-IDF news scores |
| `news_embeddings_predictions.csv` | FinBERT embedding scores |
| `eval_dataset.csv` | Joined per-model CSVs for ensemble training |
| **`final_ensemble_predictions.csv`** | **Primary UI input** — ensemble + all model columns |
| `evaluation_*.csv` | Metrics (`all`, `has_news`, `high_conf` subsets) |
| `pipeline_config.json` | Last train config (horizon, seeds, tickers) |
| `last_published.json` | Publish timestamp |

### `data/models/`

| File | Purpose |
|------|---------|
| `lstm_model.pt` | LSTM weights + scaler (+ optional seed models) |
| `nlp_baseline.joblib` / `news_tfidf.joblib` | TF-IDF + logistic pipeline |
| `news_embeddings.joblib` | FinBERT + classifier |
| `ensemble_meta.joblib` | Meta-model(s) + feature importances (Why tab) |

See [RESULTS.md](RESULTS.md) for evaluation metrics produced from these artifacts.

---

## Prediction timeline

### Labels lag prices by one session

A label for date **T** compares close(T) to close(T+1). The last *labeled* date is therefore one trading session behind the last *price* date until the next close is collected.

### Live inference

Dates after the last label are scored with saved model weights:

| Module | Role |
|--------|------|
| `src/ml/lstm_live_export.py` | LSTM rows for unlabeled price dates |
| `src/ml/news_live_export.py` | TF-IDF + FinBERT for live dates with headlines |
| `score_models.py` → `backfill_outcomes()` | Fills `actual_binary` once labels exist |

The pipeline also emits a **forward session** forecast after the latest price bar (next NYSE session using data through the prior close).

### Outcome resolution

Forecast for date **T** = move from close(T) → close(T+1).

| Viewing date T | Requires | UI state |
|----------------|----------|----------|
| T (resolved) | Close T+1 in DB + backfill | ✓ / ✗ outcome dot |
| T (latest price day) | Close T+1 not yet available | Pending ring dot |
| T (forward session) | Close T not yet available | Forecast only, pending |

`has_news` is derived from cutoff-aligned headline count in SQLite (`n_headlines > 0`).

### Non-trading days

The date scrubber lists **NYSE sessions with prediction rows** only. Weekends and exchange holidays are omitted (e.g. Fri 5/22 → Tue 5/26 across Memorial Day).

---

## Pipeline operations

### Full retrain

Recommended preset for current accuracy work:

```bash
source .venv/bin/activate
cp .env.example .env   # NEWS_API_KEY

python scripts/setup_database.py
python scripts/run_pipeline.py --preset max_v2
```

| Preset | Use case |
|--------|----------|
| `quick` | 2 tickers, smoke test |
| `balanced` | 5 tickers, 3-day horizon |
| `advanced` | All tickers, 3-day horizon, FinBERT |
| `max` | All tickers, next-day (legacy production parity) |
| **`max_v2`** | **Recommended** — FinBERT, conditional ensemble, VIX features |

Retrain **replaces** prediction CSVs. The pipeline prunes stale DB rows and old seed checkpoints.

Partial LSTM retrain only:

```bash
python scripts/run_pipeline.py --preset max_v2 \
  --skip-collect --skip-news --skip-nlp --skip-emb --skip-ensemble --skip-evaluate
python scripts/build_eval_dataset.py
python scripts/build_ensemble.py --conditional
python scripts/evaluate_predictions.py --horizon 1
```

### Daily refresh (no retrain)

```bash
python scripts/daily_update.py
```

Steps: collect prices/news → labels → `score_models.py` (live scoring + backfill) → rebuild ensemble → evaluate.

```bash
python scripts/daily_update.py --dry-run
python scripts/daily_update.py --lookback-days 90   # after long downtime
```

### Scheduled updates (launchd)

`scripts/local_cron.sh` runs `daily_update.py` then publishes to Railway. Template: `docs/local_cron.plist.example`. Default: **Mon–Fri 22:00 UTC** (~6 PM ET in EDT).

Setup (macOS Ventura+):

```bash
cp docs/local_cron.plist.example ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
# Edit WorkingDirectory and script paths
plutil -lint ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
launchctl bootout gui/$(id -u)/com.news-to-alpha.daily 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
launchctl enable gui/$(id -u)/com.news-to-alpha.daily
```

Logs: `~/Library/Logs/news-to-alpha/daily_update.log`

#### Host asleep at scheduled time

| Scenario | Behavior |
|----------|----------|
| Host asleep at cron fire time | **Job skipped** — launchd does not queue missed runs |
| Host wakes later | **No automatic catch-up** until next schedule or manual run |
| Manual catch-up after missed days | **One run suffices** — default 60-day lookback + label backfill covers all gaps |
| Downtime > 60 calendar days | Increase `--lookback-days` before publish |

```bash
bash scripts/local_cron.sh
# or:
python scripts/daily_update.py && python scripts/publish_deploy_bundle.py --target railway --service web
```

Alternative for hosts that are frequently offline: optional cloud cron (see [DEPLOY.md](DEPLOY.md)).

### Manual catch-up (step-by-step)

When models exist and only dates need advancing:

```bash
python scripts/collect_prices.py --days 14
python scripts/collect_news.py --days 14 --fill-gaps
python scripts/generate_labels.py
python scripts/score_models.py
python scripts/build_eval_dataset.py
python scripts/build_ensemble.py --conditional
python scripts/evaluate_predictions.py --horizon 1
```

Verify:

```bash
python scripts/audit_data_coverage.py
curl -s http://127.0.0.1:8000/api/data-status | python3 -m json.tool
```

### Publish to Railway

API service volume must mount at **`/data`**:

```bash
python scripts/publish_deploy_bundle.py --dry-run
python scripts/publish_deploy_bundle.py --target railway --service web
```

Requires `railway login`, `railway link`, SSH key registration (`railway ssh keys add`). Upload uses **`railway ssh`** — not `railway run` (local-only, no volume access).

---

## Local development

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

Open http://localhost:3000

---

## API freshness fields (`/api/data-status`)

| Field | Meaning |
|-------|---------|
| `latest_prediction_date` | Newest ensemble row (includes forward session) |
| `latest_resolved_prediction_date` | Newest date with `actual_binary` filled |
| `latest_price_date` | Newest price in SQLite |
| `is_current` | Predictions cover through latest price |
| `market_status` | `open` / `closed` / `pre_market` (4 PM ET cutoff) |
| `pending_reason` | `awaiting_next_close` / `awaiting_data_refresh` / `resolved` |

---

## News collection (Finnhub)

Finnhub returns **at most ~240 articles per API call**. Wide date ranges silently drop older articles.

| Mode | Command |
|------|---------|
| Daily / pipeline | `collect_news.py --days N --fill-gaps` |
| Backfill history | `--days 365 --backfill --fill-gaps --chunk-days 7` |
| Repair date range | `--start-date YYYY-MM-01 --end-date YYYY-MM-DD --fill-gaps` |

Always use `--fill-gaps` in production collection.

---

## Script reference

| Script | When to use |
|--------|-------------|
| `run_pipeline.py` | Full retrain (`--preset max_v2`) |
| `daily_update.py` | Weekday refresh without retrain |
| `score_models.py` | Live scoring + outcome backfill |
| `audit_data_coverage.py` | Coverage and consistency report |
| `publish_deploy_bundle.py` | Trim + SSH upload to Railway |
| `evaluate_predictions.py` | Regenerate [RESULTS.md](RESULTS.md) metrics |

---

## Confidence & explanations

```text
confidence = |probability − 0.5| × 2    # range 0–1
```

Measures **lean strength**, not P(correct). UI labels: Low (&lt;25%), Moderate (25–45%), Strong (≥45%).

**Why this call:** counterfactual explanation via `src/ml/ensemble_explain.py`. With conditional ensemble, headline days use a separate meta-model route.

Evaluation subsets: `all`, `has_news`, `high_conf` — see [RESULTS.md](RESULTS.md).

---

## Further reading

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — architecture and design decisions  
- [RESULTS.md](RESULTS.md) — model accuracy and findings  
- [DEPLOY.md](DEPLOY.md) — production deployment (operator doc)
