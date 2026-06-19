# Development & local testing

Everything needed to set up, retrain, verify, and run the stack locally.

---

## Prerequisites

- Python 3.11+
- Node.js 20+ (for `web/`)
- Finnhub API key ([free tier](https://finnhub.io/))

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set NEWS_API_KEY
```

Artifacts under `data/` are created by the pipeline (not committed). First-time bootstrap:

```bash
python scripts/setup_database.py
python scripts/run_pipeline.py --preset max_v2   # or --preset quick for a fast smoke test
```

---

## Full retrain

`max_v2` is the production preset: 20 tickers, 3 LSTM seeds, FinBERT encoder, conditional ensemble.

```bash
source .venv/bin/activate

# data already in the DB — skip fetch for faster iteration
python scripts/run_pipeline.py --preset max_v2 --skip-collect --skip-news

# from scratch (includes prices, news, FinBERT sentiment scoring)
python scripts/run_pipeline.py --preset max_v2
```

**Stages:** labels → split → LSTM → TF-IDF news → FinBERT embeddings → volatility →
eval dataset → conditional ensemble → `evaluate_predictions.py`. When news is collected
(not skipped), FinBERT sentiment is scored right after `collect_news`.

**Duration:** ~30–90 min on CPU (FinBERT + 3× LSTM seeds dominate).

> If you skip news during a retrain, remember to collect it separately or the latest
> sessions will have no headlines. See [DATA.md](DATA.md) for collection commands.

---

## Run the API and UI

Two terminals from the repository root.

**API (Flask)**

```bash
source .venv/bin/activate
python app/server.py --port 8000
```

Port 8000 avoids the macOS AirPlay conflict on 5000, and matches the UI default
(`web/.env.local` → `API_BASE_URL=http://127.0.0.1:8000`).

**UI (Next.js)**

```bash
cd web
cp .env.example .env.local    # API_BASE_URL=http://127.0.0.1:8000
npm install && npm run dev
```

Open http://localhost:3000 — the UI calls Flask via `API_BASE_URL`.

| Route | Shows |
|-------|-------|
| `/` | Markets grid — direction call + expected-move band |
| `/t/[symbol]` | Ticker detail: hero call, headlines, Why / Advanced tabs |
| `/status` | Data freshness + evaluation summary |

---

## Refresh predictions (no retrain)

After market close, or to verify the incremental pipeline:

```bash
python scripts/daily_update.py --dry-run    # preview window, no writes
python scripts/daily_update.py
python scripts/audit_data_coverage.py
```

Steps: collect prices/news → score sentiment → labels → live scoring → rebuild
ensemble → evaluate. See [DATA.md § Daily refresh](DATA.md) for incremental and
catch-up options.

| Step | Good signs | Warning signs |
|------|------------|---------------|
| Header | `gap_days` 0–2 when current | `gap_days` > 7 — stale data |
| `collect_prices` | 0–22 rows added | failed tickers listed |
| `collect_news` | some new articles | 0 articles every day — check API key |
| `score_models` | `Appended N live rows` | traceback in FinBERT / LSTM load |
| Footer | `DAILY UPDATE COMPLETE` | exit code 1 |

Run metadata: `data/processed/pipeline_config.json` → `last_daily_update`.

---

## Verify after a retrain or change

```bash
# unit tests (a few date/mock cases may fail depending on the calendar)
pytest tests/unit -q

# LSTM calibration guard (should pass — protects against probability collapse)
pytest tests/unit/test_lstm_calibration.py -q

# evaluation summary
python scripts/evaluate_predictions.py --horizon 1
cat data/processed/evaluation_summary.txt

# walk-forward harness (multi-fold, less optimistic than the single split)
python scripts/walk_forward_eval.py --all-targets

# LSTM probabilities should be many distinct values, not a handful
python -c "import pandas as pd; p=pd.read_csv('data/processed/price_predictions.csv'); print('distinct proba:', p['financial_pred_proba'].nunique(), 'std:', round(p['financial_pred_proba'].std(),4))"
```

Report the **`news_scored`** subset for news/ensemble numbers (out-of-sample
news window). See [RESULTS.md](RESULTS.md).

### API smoke tests

With Flask on port 8000:

```bash
curl -s http://127.0.0.1:8000/api/data-status | python3 -m json.tool
curl -s "http://127.0.0.1:8000/api/ticker?ticker=AAPL&model=ensemble" | python3 -m json.tool
curl -s "http://127.0.0.1:8000/api/rationale?ticker=AAPL&date=<latest>" | python3 -m json.tool
```

Pick `<latest>` from the max `prediction_date` in
`data/processed/final_ensemble_predictions.csv`.

Checklist:
1. `/api/ticker` returns non-null `expected_move_pct`, `forecast_low`, `forecast_high`.
2. News probas in `per_model` are not exactly 0.5 when headlines exist.
3. `/api/rationale` includes `baseline_proba` and `drivers` that sum to the final call.
4. UI markets grid shows all 20 tickers; resolved dates show ✓/✗, the latest shows a pending ring.

Restart Flask after API changes or a retrain.

---

## Build & deploy checks

```bash
pytest tests/unit -q
cd web && npm run build
python scripts/publish_deploy_bundle.py --dry-run
```

Inference artifacts (CSVs, models, SQLite) publish to a remote `/data` volume
separately from app deploys:

```bash
python scripts/publish_deploy_bundle.py --target railway --service web
```

Requires the Railway CLI, SSH keys, and a linked project (see [DATA.md § Publish](DATA.md)).
Scheduled weekday refreshes run on **GitHub Actions** (pull → `daily_update.py` → publish).
Verify CI auth locally before debugging Actions:

```bash
export RAILWAY_API_TOKEN="<Account token from railway.com/account/tokens>"
export RAILWAY_PROJECT_ID="91bd0f77-9c80-416e-8f60-8409ae0f0927"
export RAILWAY_SERVICE="web"
export RAILWAY_ENVIRONMENT="production"
export RAILWAY_SSH_KEY_PATH="$HOME/.ssh/railway_github_actions"
bash scripts/verify_railway_ci.sh
```

GitHub secret `RAILWAY_API_TOKEN` must be the full Account token (No workspace).
The CLI reads it as **`RAILWAY_API_TOKEN`**, not `RAILWAY_TOKEN` — do not set both.

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| UI empty / 503 | Flask running; `API_BASE_URL` in `web/.env.local` points to `:8000` |
| Stale predictions | Run `daily_update.py`; confirm `/api/data-status` |
| Latest session pending | Expected until the T+1 close is collected and backfilled |
| Recent dates missing tickers | News or prices not collected — re-run `daily_update.py` |
| `ModuleNotFoundError: src` | Activate venv; run commands from the repo root |
| Charts blank / Why tab empty | Stale Flask or NaN in JSON — restart `app/server.py` |
| GitHub Action: Unauthorized / login | Regenerate Account token; update `RAILWAY_API_TOKEN` secret; run `scripts/verify_railway_ci.sh` |

More detail: [DATA.md](DATA.md), [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).
