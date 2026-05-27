# Deploy: UI + API

Step-by-step deployment for the **Next.js UI (Vercel)** and **Flask API (Railway)**.

> Gitignored local doc — not in the public repo.

---

## Overview

```
Mac (train + daily_update)  →  publish_deploy_bundle.py  →  Railway /data volume
                                                              ↑
Vercel (web/)  ──API_BASE_URL──→  Flask API (Dockerfile.inference)
```

---

## Part A — Train & bundle (Mac)

```bash
source .venv/bin/activate
python scripts/run_pipeline.py --preset max
python scripts/publish_deploy_bundle.py --target local --output deploy_bundle/
```

---

## Part B — Railway API

1. GitHub → Railway project **news-to-alpha** → service **web**
2. **Variables:**
   - `NEWS_API_KEY` — Finnhub key
   - `RAILWAY_DOCKERFILE_PATH` = `Dockerfile.inference`
   - **Do not set `PORT`** — Railway injects it
3. **Volume** on service **web**: mount path **`/data`**, 1 GB
4. **Networking:** Generate domain → note URL
5. Redeploy after Dockerfile / Procfile fixes

The Docker image sets `DATA_DIR=/data`, `DATABASE_PATH=/data/database.db`, etc. automatically.

---

## Part C — Upload bundle

```bash
railway login
railway link   # project news-to-alpha → service web
python scripts/publish_deploy_bundle.py --target railway
```

Smoke test:

```bash
curl -s https://YOUR-URL.up.railway.app/healthz
curl -s https://YOUR-URL.up.railway.app/api/data-status
```

---

## Part D — Vercel UI

1. Import repo → **Root Directory:** `web`
2. **Env:** `API_BASE_URL=https://YOUR-URL.up.railway.app`
3. Deploy

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `$PORT` not valid | Remove `PORT` variable; Procfile uses `sh -c` |
| `mkdir: /app: Read-only` | Volume must be **`/data`**, not `/app/data` |
| Upload fails | Volume mounted at `/data`; service running; `railway link` to **web** |
| Empty predictions | Run publish script; check `/api/data-status` |
