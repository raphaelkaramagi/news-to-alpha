# Deploying News-to-Alpha (API)

> **UI deploy:** The public dashboard is the Next.js app in `web/` (Vercel). See **[DEPLOY_UI.md](DEPLOY_UI.md)** for the full stack (Mac train → Railway API → Vercel UI).
>
> This document covers the **Flask JSON API** only. Use **`Dockerfile.inference`** for production (lean image, `INFERENCE_ONLY=true`). The legacy Flask HTML UI has been removed.

The API runs as a single Flask service behind gunicorn. Tested on **Railway**, **Render**, and **Fly.io** (long-lived processes + persistent volume).

---

## 1. Files that make deployment possible

| File | Role |
|------|------|
| `Dockerfile.inference` | **Production** lean image (Railway) |
| `Dockerfile` | Full image with training deps (local / one-off) |
| `Procfile` | Heroku/Railway-style process definition |
| `.dockerignore` | Keeps image lean |
| `requirements-inference.txt` | Lean deps for API-only deploy |
| `requirements.txt` | Full training stack (Mac) |
| `app/server.py` | JSON API; `/healthz` for platform checks |

---

## 2. Environment variables

| Name | Required | Description |
|------|----------|-------------|
| `NEWS_API_KEY` | yes | Finnhub API key |
| `INFERENCE_ONLY` | prod | `true` — disables train/run/reset HTTP routes |
| `DATABASE_PATH` | prod | e.g. `/data/database.db` |
| `PROCESSED_DATA_DIR` | prod | e.g. `/data/processed` |
| `MODELS_DIR` | prod | e.g. `/data/models` |
| `HF_HOME` | no | HuggingFace cache; default `/data/hf-cache` in Docker |
| `PORT` | auto | Injected by platform; gunicorn binds to it |
| `ENABLE_CLOUD_INFER` | no | Optional Railway cron infer (see personal deploy doc) |

---

## 3. Persistent storage

Mount a volume at `/data` so SQLite, CSVs, models, and HF weights survive redeploys.

Recommended layout on the volume:

```
/data/
  database.db
  processed/final_ensemble_predictions.csv
  processed/evaluation_*.csv
  models/*.joblib, *.pt
  hf-cache/
```

Point env vars at these paths (see table above).

---

## 4. Railway

1. New Project → Deploy from GitHub.
2. **Settings → Dockerfile Path:** `Dockerfile.inference`
3. **Volumes:** mount `/data` (1 GB).
4. Set variables (`NEWS_API_KEY`, `INFERENCE_ONLY=true`, paths under `/data`).
5. Deploy. Upload bundle after first deploy:

   ```bash
   python scripts/publish_deploy_bundle.py --target railway
   ```

---

## 5. Render / Fly.io

Same pattern: Docker build from `Dockerfile.inference`, disk/volume at `/data`, env vars as above. See historical notes in git history for `fly.toml` mount examples.

---

## 6. One-time initial data load

After first deploy, `/api/data-status` shows no predictions until you upload a bundle:

```bash
python scripts/run_pipeline.py --preset balanced
python scripts/publish_deploy_bundle.py --target railway
```

---

## 7. Smoke tests after deploy

```bash
BASE=https://<your-app>.up.railway.app

curl -s $BASE/healthz
curl -s $BASE/api/data-status
curl -s $BASE/api/metrics
curl -s "$BASE/api/ticker?ticker=AAPL&model=ensemble"
curl -s "$BASE/api/dates?ticker=AAPL"
curl -s "$BASE/api/history?ticker=AAPL&window=30"
curl -s "$BASE/api/rationale?ticker=AAPL&date=2026-05-22"
curl -s "$BASE/api/markets-overview?window=30"
curl -s "$BASE/api/accuracy-summary?ticker=ALL&window=30"
```

All should return `200` with JSON. If `/healthz` reports `predictions_csv: false`, publish the bundle ([DATA.md](DATA.md)).

### API surface (read-only in production)

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | API info JSON |
| GET | `/healthz` | Liveness + predictions CSV check |
| GET | `/api/data-status` | Freshness, row counts |
| GET | `/api/ticker` | Prediction for ticker/model/date |
| GET | `/api/history` | Prices + per-day predictions (charts) |
| GET | `/api/dates` | Dates with predictions (date scrubber) |
| GET | `/api/headlines` | Headline cards |
| GET | `/api/rationale` | Ensemble explanation (counterfactual + votes) |
| GET | `/api/last-resolved` | Recent resolved calls |
| GET | `/api/accuracy-summary` | Windowed hit rate |
| GET | `/api/accuracy-trace` | Rolling accuracy series |
| GET | `/api/markets-overview` | All-ticker price index + daily accuracy |
| GET | `/api/conviction` | Accuracy by confidence bucket |
| GET | `/api/metrics` | Overall + per-ticker evaluation |

When `INFERENCE_ONLY=true`, mutation routes (`POST /api/run`, `/api/train`, `/api/reset`, etc.) return **403**. Training is CLI-only on your Mac.

---

## 8. Next.js frontend (Vercel)

Separate deploy from Flask:

1. Vercel project with **Root Directory** = `web`.
2. Env: `API_BASE_URL` = public Flask origin (no trailing slash).
3. Next.js route handlers (`web/app/api/*`) proxy to Flask.

Local dev: `cd web && cp .env.example .env.local` → `API_BASE_URL=http://127.0.0.1:8000`. See [web/README.md](../web/README.md).

---

## 9. Scaling

- Keep gunicorn **`--workers 1`** so the in-process job lock works. Use `--threads` for more read throughput.
- Heavy training belongs on your Mac, not the API container.
