# Deploy: UI + API

Step-by-step deployment for the **Next.js UI (Vercel)** and **Flask API (Railway)**.

Training stays on your Mac. See [DATA.md](DATA.md) for how to refresh predictions before publishing.

---

## Overview

```
Mac (train + daily_update)  →  publish_deploy_bundle.py  →  Railway /data volume
                                                              ↑
Vercel (web/)  ──API_BASE_URL──→  Flask API (Dockerfile.inference)
```

Estimated cost: **~$8–12/mo** (Vercel Hobby + Railway Hobby, Mac cron only).

---

## Part A — Train & bundle (Mac, one-time)

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python scripts/run_pipeline.py --preset balanced
python scripts/publish_deploy_bundle.py --dry-run
python scripts/publish_deploy_bundle.py --target local --output deploy_bundle/
```

---

## Part B — Railway API

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub.
2. **Settings → Build:** Dockerfile path = **`Dockerfile.inference`** (not the full `Dockerfile`).
3. **Volumes:** mount `/data` (1 GB).
4. **Variables:**

   | Name | Value |
   |------|-------|
   | `NEWS_API_KEY` | Finnhub key |
   | `INFERENCE_ONLY` | `true` |
   | `DATABASE_PATH` | `/data/database.db` |
   | `PROCESSED_DATA_DIR` | `/data/processed` |
   | `MODELS_DIR` | `/data/models` |
   | `HF_HOME` | `/data/hf-cache` |

5. Deploy → note URL: `https://<app>.up.railway.app`
6. Upload bundle: `python scripts/publish_deploy_bundle.py --target railway`
7. Smoke test:

   ```bash
   curl -s https://<app>.up.railway.app/healthz
   curl -s https://<app>.up.railway.app/api/data-status
   curl -s "https://<app>.up.railway.app/api/rationale?ticker=AAPL&date=2026-05-22" | head -c 300
   ```

No Railway redeploy needed for data updates — only the volume contents change.

---

## Part C — Vercel UI

1. [vercel.com](https://vercel.com) → Import repo.
2. **Root Directory:** `web` (required — there is no root `package.json`).
3. **Environment:** `API_BASE_URL=https://<app>.up.railway.app` (no trailing slash).
4. Deploy → open `https://<project>.vercel.app`

Smoke: home page loads with ticker grid + overview chart, `/t/AAPL` shows Why tab, `/status` shows freshness.

---

## Part D — Keep data current

On your Mac (weekdays after market close):

```bash
python scripts/daily_update.py
python scripts/publish_deploy_bundle.py --target railway
```

Optional: copy `docs/local_cron.plist.example` to `~/Library/LaunchAgents/` (gitignored template — edit paths locally).

Occasional retrain:

```bash
python scripts/run_pipeline.py --preset balanced
python scripts/publish_deploy_bundle.py --target railway
```

---

## Custom domain (optional)

Vercel → Domains → add `your-domain.com` → DNS CNAME to Vercel.

Railway URL stays backend-only.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| UI 503 | Check `API_BASE_URL` on Vercel; Railway service running |
| `npm run dev` ENOENT at repo root | Run from `web/`: `cd web && npm run dev` |
| Stale dates / missing Why tab | Run `daily_update` + publish; **restart Flask** locally after code pulls |
| Port 8000 in use | `lsof -nP -iTCP:8000 -sTCP:LISTEN` → kill old Flask, restart |
| Empty headlines | Ensure news collected; check date scrubber |
| Chart missing | Verify `/api/history?ticker=X`; NaN-safe JSON requires current Flask |
| Markets overview empty | Restart Flask (`/api/markets-overview`); Next.js has a fallback aggregator |
| Outcome dots missing | Hard-refresh browser; scrub to a **resolved** date (latest session is pending) |

See [DATA.md](DATA.md) for detailed update commands. Personal step-by-step deploy: `docs/DEPLOY_VERCEL.md` (gitignored, local only).
