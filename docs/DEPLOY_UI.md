# Deploy: UI + API

Full deployment runbook for **Stock Price and Sentiment Predictor**  
(Vercel UI + Railway Flask API). Gitignored — local only.

---

## Live URLs (your setup)

| Layer | URL |
|-------|-----|
| **UI (Vercel)** | https://stock.raphaelkaramagi.com |
| **API (Railway)** | https://web-production-ac596.up.railway.app |

Vercel env: `API_BASE_URL=https://web-production-ac596.up.railway.app` (no trailing slash).

---

## Architecture

```
Mac (train + daily_update)  →  publish_deploy_bundle.py  →  Railway /data volume
                                                              ↑
Vercel (web/)  ──API_BASE_URL──→  Flask API (Dockerfile.inference)
```

Train locally. Upload artifacts only. No ML on cloud.

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
3. **Volume** on service **web**: mount path **`/data`**, 1 GB is enough
4. **Networking:** Generate domain → note URL
5. Redeploy after Dockerfile / Procfile changes

`Dockerfile.inference` sets `DATA_DIR=/data`, `DATABASE_PATH=/data/database.db`, etc.

---

## Part C — Upload bundle

Service must be **Online**. Volume mount: **`/data`**.

**One-time SSH setup** (uploads use `railway ssh`, not `railway run`):

```bash
railway login
railway link   # project news-to-alpha → service web
railway ssh keys add
railway ssh config -i ~/.ssh/id_ed25519
# If "Host key verification failed":
ssh-keyscan ssh.railway.com >> ~/.ssh/known_hosts
```

```bash
railway ssh -- ls -la /data   # should list processed/, models/, not "Read-only"

python scripts/publish_deploy_bundle.py --target railway --service web
```

If permission errors: add `RAILWAY_RUN_UID=0` on the web service.

**Smoke test (Railway direct):**

```bash
curl -s https://web-production-ac596.up.railway.app/healthz
curl -s https://web-production-ac596.up.railway.app/api/data-status
```

**Smoke test (Vercel proxy):**

```bash
curl -s https://stock.raphaelkaramagi.com/api/data-status
curl -s https://stock.raphaelkaramagi.com/api/healthz   # proxies Flask /healthz
```

---

## Part D — Vercel UI

1. [vercel.com](https://vercel.com) → Import **news-to-alpha** repo
2. **Root Directory:** `web`
3. **Env:** `API_BASE_URL=https://web-production-ac596.up.railway.app`
4. Deploy
5. **Domains:** add `stock.raphaelkaramagi.com` → CNAME to Vercel DNS

Icons and tab title are in `web/app/layout.tsx` + `web/app/icon.png`.

---

## Part E — Mac daily cron (optional)

```bash
cp docs/local_cron.plist.example ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
# Edit paths, then:
launchctl load ~/Library/LaunchAgents/com.news-to-alpha.daily.plist
```

Runs `daily_update.py` + `publish_deploy_bundle.py --target railway` on weekdays.

---

## Cost (already near-minimal)

| Item | Cost | Notes |
|------|------|-------|
| Vercel Hobby | $0 | Static/SSR UI, custom domain included |
| Railway subscription | $5/mo | Required for volumes |
| Railway web service | ~$3–8/mo | 1 worker gunicorn, read-only API |
| Volume 1 GB | ~$0.25/mo | Bundle is ~10 MB — don't oversize |
| Finnhub | $0 | Free tier |
| **Total** | **~$8–13/mo** | |

**Do not enable** unless needed:
- `ENABLE_CLOUD_INFER=true` (+ CPU on Railway)
- Railway cron for daily infer (Mac cron is free)
- Larger volume / multiple replicas

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `$PORT` crash on deploy | Remove manual `PORT` var; Procfile uses `sh -c` |
| `mkdir: /data: Read-only` | Used `railway run` by mistake — use `railway ssh` |
| `/api/healthz` 404 on Vercel | Fixed: proxy hits Flask `/healthz` (redeploy UI) |
| Empty predictions | Run publish script; check Railway `/api/data-status` |
| Upload fails | SSH keys registered; service Online; volume at `/data` |

See also: `docs/PERSONAL_FULL_GUIDE.md` for full architecture context.
