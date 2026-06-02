# Development guide

Setup and verification for contributors running the stack locally.

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
python scripts/run_pipeline.py --preset max_v2   # or --preset quick for a smoke test
```

---

## Run the API and UI

Use two terminals from the repository root.

**API (Flask)**

```bash
source .venv/bin/activate
python app/server.py --port 8000
```

Port 8000 avoids conflicts with macOS AirPlay on 5000.

**UI (Next.js)**

```bash
cd web
cp .env.example .env.local    # API_BASE_URL=http://127.0.0.1:8000
npm install && npm run dev
```

Open http://localhost:3000 — the UI proxies `/api/*` to Flask via `API_BASE_URL`.

---

## Refresh predictions (no retrain)

After market close or to verify the daily pipeline:

```bash
source .venv/bin/activate
python scripts/daily_update.py --dry-run
python scripts/daily_update.py
python scripts/audit_data_coverage.py
```

See [DATA.md § Daily refresh](DATA.md) for incremental collection and catch-up options.

---

## Smoke tests

With Flask on port 8000:

```bash
curl -s http://127.0.0.1:8000/healthz | python3 -m json.tool
curl -s http://127.0.0.1:8000/api/data-status | python3 -m json.tool
curl -s "http://127.0.0.1:8000/api/ticker?ticker=AAPL&model=ensemble" | python3 -m json.tool
curl -s "http://127.0.0.1:8000/api/rationale?ticker=AAPL&date=2026-06-01" | python3 -m json.tool
```

Verify `/api/ticker` includes non-null `expected_move_pct`, `forecast_low`, `forecast_high` for live rows. Live news probas in `per_model` should not be exactly 0.5 when headlines exist.

After retrain, regenerate metrics (report **`news_scored`** for honest news/ensemble numbers):

```bash
python scripts/evaluate_predictions.py --horizon 1
cat data/processed/evaluation_summary.txt
```

Restart Flask after API changes or after retraining.

### Daily update checklist

After market close, or to verify the incremental pipeline:

```bash
python scripts/daily_update.py --dry-run   # preview window, no writes
python scripts/daily_update.py
python scripts/audit_data_coverage.py
```

| Step | Good signs | Warning signs |
|------|------------|---------------|
| Header | `gap_days` 0–2 when data is current | `gap_days` > 7 — stale data |
| `collect_prices` | 0–22 rows added | Failed tickers listed |
| `collect_news` | Some new articles | 0 articles every day — check API key |
| `score_models` | `Appended N live rows` | Traceback in FinBERT / LSTM load |
| Footer | `DAILY UPDATE COMPLETE` | Exit code 1 |

Run metadata: `data/processed/pipeline_config.json` → `last_daily_update`.

See also [LOCAL_TESTING.md](LOCAL_TESTING.md) for UI walkthrough and troubleshooting tables.

---

## Automated tests

```bash
pytest tests/unit -q
cd web && npm run build
python scripts/publish_deploy_bundle.py --dry-run
```

---

## Production data sync

Inference artifacts (CSVs, models, SQLite) are published to a remote `/data` volume separately from application deploys:

```bash
python scripts/publish_deploy_bundle.py --target railway --service web
```

Requires Railway CLI, SSH keys, and a linked project. See [DATA.md § Publish](DATA.md).

Scheduled weekday refreshes can run via the GitHub Actions workflow in `.github/workflows/daily-update.yml` (repository secrets required — configure in your fork or deployment environment).

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| UI empty / 503 | Flask running; `API_BASE_URL` in `web/.env.local` |
| Stale predictions | Run `daily_update.py`; confirm `/api/data-status` |
| `ModuleNotFoundError: src` | Activate venv; run commands from repo root |
| Latest date pending | Expected until T+1 close is collected and backfilled |

More detail: [DATA.md](DATA.md), [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).
