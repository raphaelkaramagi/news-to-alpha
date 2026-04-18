# Deploying News-to-Alpha

This app is designed to run as a single Flask service behind gunicorn. It's
been tested on **Railway**, **Render**, and **Fly.io** (all of which support
long-lived processes and a persistent volume).

---

## 1. Files that make deployment possible

- `app/static/icons/` — favicons and `site.webmanifest` (tab icon + “Add to Home Screen” on mobile). Same URLs work on **Render** and any custom domain (e.g. `stock.raphaelkaramagi.com`): Flask serves `/static/...` from the container image. This is independent of **Vercel** (your root/www/ASL projects use Vercel’s own icons there).
- `Dockerfile` — production image, Python 3.11 slim, pins `HF_HOME` so the
  FinBERT and MiniLM caches survive restarts.
- `Procfile` — Heroku/Railway-style process definition.
- `.dockerignore` — keeps image lean (no `data/raw`, no tests, no notebooks).
- `requirements.txt` — now includes `gunicorn` and `transformers`.
- `app/server.py` exposes `/healthz` for platform health checks.

## 2. Environment variables

| Name              | Required | Description                                              |
|-------------------|----------|----------------------------------------------------------|
| `NEWS_API_KEY`    | yes      | Finnhub API key (used by `src/data_collection/news_collector.py`). |
| `PORT`            | auto     | Injected by the platform; `gunicorn` binds to it.        |
| `HF_HOME`         | no       | Defaults to `/data/hf-cache` in the Dockerfile.          |
| `PYTHONUNBUFFERED`| no       | Already `1` in the Dockerfile.                           |

## 3. Persistent storage

You want a volume mounted at `/data` so that:

- FinBERT / MiniLM weights (downloaded the first time) aren't re-fetched on
  every cold start.
- The SQLite DB, `data/processed/*.csv`, and `data/models/*.joblib` survive
  redeploys.

The simplest pattern: mount a single volume at `/data` and symlink the
sub-dirs from `/app/data/...` to it (or point `DATABASE_PATH` etc. via env
vars — see `src/config.py`).

## 4. Railway (one-click)

1. `railway init` in the repo.
2. Add a volume: Settings → Volumes → mount at `/data` (1 GB is plenty for
   starters; FinBERT weights alone are ~440 MB).
3. Set `NEWS_API_KEY` under Variables.
4. Deploy. Railway auto-detects the `Dockerfile` and injects `PORT`.

## 5. Render

1. New → Web Service → connect the repo.
2. Environment: **Docker**.
3. Add a **Disk** (size 1 GB, mount path `/data`).
4. Set env vars (`NEWS_API_KEY`).
5. Create service. Render will invoke the Dockerfile's `CMD`.

## 6. Fly.io

```bash
fly launch --dockerfile Dockerfile --no-deploy
fly volumes create nta_data --region <region> --size 1
fly deploy
```

Add to `fly.toml`:

```toml
[mounts]
  source = "nta_data"
  destination = "/data"
```

## 7. One-time initial data load

After the first deploy, the landing page (`/`) will show that no predictions
exist yet. Two ways to populate:

- Click a preset (`quick` / `balanced` / `advanced`) and **Run pipeline** —
  the app calls `POST /api/run` which shells out to
  `scripts/run_pipeline.py` with your chosen config.
- Or SSH into the container and run it manually:
  ```bash
  railway run python scripts/run_pipeline.py --preset balanced
  ```

Once the run finishes, `/dashboard` becomes the default landing surface
(accessible from the top nav, or directly at `/dashboard`).

## 8. Smoke tests after deploy

### Tab icon / favicon looks wrong or never updates

1. **Verify the files exist** (replace host with your Render URL or `stock.raphaelkaramagi.com`):
   ```bash
   curl -sI https://YOUR_HOST/static/icons/favicon-32x32.png
   curl -sI https://YOUR_HOST/favicon.ico
   ```
   Expect **`200`** and `content-type: image/png`. If you get **404**, the deploy image may be missing `app/static/` — confirm the commit was pushed and Render rebuilt.
2. **Aggressive browser cache**: favicons are cached hard. Try a **private window**, or clear site data for that origin, or bump `ASSET_VERSION` in `app/server.py` after icon changes (query string `?v=` forces a fresh fetch for linked assets).
3. **Local `curl` to port 5000 on macOS**: often hits **AirPlay Receiver**, not Flask (`Server: AirTunes` / 403). Run the app on **8000** (or turn off AirPlay Receiver in System Settings → AirDrop & Handoff) and test `http://127.0.0.1:8000/favicon.ico`.

```bash
curl -s https://<your-app>.up.railway.app/healthz
curl -s https://<your-app>.up.railway.app/api/presets
curl -s https://<your-app>.up.railway.app/api/metrics
curl -s https://<your-app>.up.railway.app/api/conviction
curl -s "https://<your-app>.up.railway.app/api/ticker?ticker=AAPL&model=ensemble"
curl -s "https://<your-app>.up.railway.app/api/dates?ticker=AAPL"
curl -s "https://<your-app>.up.railway.app/api/accuracy-trace?ticker=AAPL&window=30"
```

All of them should return `200` with JSON. If `/healthz` reports
`predictions_csv: false`, run the pipeline once (either via the landing
page or via SSH).

### Full API surface

| Method | Route                          | Purpose                                         |
|--------|--------------------------------|-------------------------------------------------|
| GET    | `/`                            | Landing / configure                             |
| GET    | `/dashboard`                   | Interactive dashboard                           |
| GET    | `/admin`                       | Per-model retrain / reset UI                    |
| GET    | `/healthz`                     | Liveness                                        |
| GET    | `/api/presets`                 | Available pipeline presets                      |
| POST   | `/api/run`                     | Kick off the full pipeline (background job)     |
| POST   | `/api/train`                   | Per-model retrain (`lstm` / `tfidf` / `embeddings` / `ensemble` / `all`) |
| POST   | `/api/reset`                   | Remove a model's artifacts                      |
| GET    | `/api/jobs`                    | Current + recent background jobs                |
| GET    | `/api/ticker?ticker=&model=`   | Latest prediction for a ticker/model            |
| GET    | `/api/history?ticker=&window=` | Prices + per-day predictions                    |
| GET    | `/api/dates?ticker=`           | Dates that have a prediction (for the slider)   |
| GET    | `/api/headlines?ticker=&date=` | Full headline cards (title, url, sentiment)     |
| GET    | `/api/rationale?ticker=&date=` | Meta-feature contributions for a specific call  |
| GET    | `/api/accuracy-trace?ticker=&window=` | Rolling-accuracy series for the sparkline |
| GET    | `/api/conviction`              | `evaluation_by_confidence.csv` as JSON          |
| GET    | `/api/metrics`                 | Overall + per-ticker evaluation metrics         |

## 9. Scaling

- Keep `--workers 1` in `gunicorn` so the single-job lock in `app/jobs.py`
  actually prevents concurrent retrains. If you need more web throughput,
  use `--threads` instead.
- For heavier training, move it off the web process (e.g. Railway cron job
  or a separate worker) and drop the in-process job queue.
