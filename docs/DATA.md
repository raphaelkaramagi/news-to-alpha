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
| `volatility_predictions.csv` | Expected next-day |return| (`expected_move_pct`) |
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
| `volatility_model.joblib` | Expected-move regressor |
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

The pipeline scores **one live row per latest price session** (close T → close T+1). Predictions refresh on batch `daily_update` / cron only — not dynamically as news arrives.

### Outcome resolution

Forecast for date **T** = move from close(T) → close(T+1). The UI shows both closes when resolved (`price_context` on `/api/ticker`).

| Viewing date T | Requires | UI state |
|----------------|----------|----------|
| T (resolved) | Close T+1 in DB + backfill | ✓ / ✗ outcome mark |
| T (latest price day) | Close T+1 not yet available | Pending (open ring) |

`has_news` is derived from cutoff-aligned headline count in SQLite (`n_headlines > 0`).

### Price collection (yfinance)

`PriceCollector` uses yfinance’s **`end` date as exclusive**. The collector adds one calendar day to `end_date` so the same-day session bar is included after market close. Without this, weekday `daily_update` runs could ingest prices through yesterday only and leave the latest forecast stuck **pending** even after 4 PM ET.

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
| **`max_v2`** | **Default full train** — FinBERT embeddings, conditional ensemble, VIX features, 13-feature meta schema |

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

**Incremental by default:** queries SQLite for the latest price date per ticker, then collects only `(gap + buffer)` calendar days (minimum 7, maximum `--lookback-days` default 60). Avoids re-fetching 60 days of news when data is current.

```bash
python scripts/daily_update.py --dry-run          # preview computed window
python scripts/daily_update.py --full-lookback    # fixed 60-day window (legacy)
python scripts/daily_update.py --lookback-days 90 # after long downtime
```

Steps: collect prices/news → labels → `score_models.py` (LSTM + news + volatility live scoring + backfill) → rebuild ensemble → evaluate.

Run metadata saved to `data/processed/pipeline_config.json` → `last_daily_update`.

Daily runs use all **20 canonical tickers** and **horizon 1**, regardless of older values stored in `pipeline_config.json`. NLP scoring uses **`--incremental`** by default (live rows only) to avoid full FinBERT rescoring on constrained hosts.

### Scheduled updates

| Approach | Description |
|----------|-------------|
| **GitHub Actions** | [`.github/workflows/daily-update.yml`](../.github/workflows/daily-update.yml) runs `daily_update.py` on the production host via `railway ssh` (Mon–Fri 22:00 UTC). Requires repository secrets (`RAILWAY_API_TOKEN`, SSH key, project/service IDs). |
| **Manual / CI** | `python scripts/daily_update.py` on any host with access to `/data`, or publish from a training host via `publish_deploy_bundle.py`. |
| **Host cron** | Wrap `daily_update.py` (+ optional publish) in cron or systemd on a machine that stays online after market close. |

When the job runs **inside** the Railway container on `/data`, no separate publish step is needed — outputs are written where the API reads them.

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

### Pull data from Railway

After production daily updates, sync the volume back to your training host:

```bash
python scripts/pull_railway_data.py              # DB + key CSVs/models
python scripts/pull_railway_data.py --db-only    # database only
python scripts/generate_labels.py
```

Then retrain locally with `--skip-collect --skip-news` if the DB is already current.

### Publish to Railway

API service volume must mount at **`/data`**:

```bash
python scripts/publish_deploy_bundle.py --dry-run
python scripts/publish_deploy_bundle.py --target railway --service web
```

Requires `railway login`, `railway link`, SSH key registration (`railway ssh keys add`). Upload uses **`railway ssh`** — not `railway run` (local-only, no volume access).

---

## Ticker API fields (`/api/ticker`)

| Field | Meaning |
|-------|---------|
| `price_context.start_close` / `end_close` | Closes for session T and T+1 when available |
| `price_context.validation_basis` | Always `close_to_close` for next-day horizon |
| `forecast_date` | Target close session (T+1), from `price_context.end_close_date` |

Resolved history strip and accuracy panels use **`/api/accuracy-summary?ticker=&window=7|30|90`** so counts match the chart window.

---

## API freshness fields (`/api/data-status`)

| Field | Meaning |
|-------|---------|
| `latest_prediction_date` | Max date in CSV (legacy forward rows pruned on next score) |
| `primary_prediction_date` | **Default UI date** — latest price session with predictions |
| `latest_resolved_prediction_date` | Newest date with `actual_binary` filled |
| `latest_price_date` | Newest price in SQLite |
| `expected_latest_prediction_date` | Same as `latest_price_date` |
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

Evaluation subsets: `all`, `has_news`, **`news_scored`**, `news_oos`, `high_conf` — see [RESULTS.md](RESULTS.md). Report **`news_scored`** for honest news/ensemble accuracy (out-of-sample news window, n≈399).

---

## Further reading

- [DEVELOPMENT.md](DEVELOPMENT.md) — local setup and verification  
- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — architecture and design decisions  
- [RESULTS.md](RESULTS.md) — model accuracy and findings
