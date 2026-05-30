# Automated daily updates (GitHub Actions → Railway)

Run `daily_update.py` on the Railway `/data` volume **without** your laptop. No `publish_deploy_bundle.py` step — the job writes CSVs and SQLite where the API already reads them.

**Schedule:** Mon–Fri **22:00 UTC** (~6:00 PM ET in EDT).  
**Cost:** Uses a few minutes of CPU/RAM on your existing `web` service; typically stays within the Hobby **$5** included usage.

---

## One-time setup (your machine)

### 1. Deploy the inference image with `yfinance`

Push `main` (or merge the PR) so Railway rebuilds from `Dockerfile.inference`. Confirm the deploy succeeds in the Railway dashboard.

### 2. Confirm `NEWS_API_KEY` on Railway

In Railway → project **news-to-alpha** → service **web** → **Variables**:

| Variable | Required |
|----------|----------|
| `NEWS_API_KEY` | Yes (Finnhub) |
| `DATABASE_PATH` | `/data/database.db` |
| `PROCESSED_DATA_DIR` | `/data/processed` |
| `MODELS_DIR` | `/data/models` |
| `INFERENCE_ONLY` | `true` |

The daily job runs inside this container and uses the same variables.

### 3. Create an SSH key for GitHub Actions only

```bash
ssh-keygen -t ed25519 -f ~/.ssh/railway_github_actions -N "" -C "github-actions-news-to-alpha"
```

Do **not** reuse your personal laptop key unless you intend to.

### 4. Register the public key with Railway

```bash
railway login
railway link   # select news-to-alpha / web / production
railway ssh keys add --key ~/.ssh/railway_github_actions.pub --name "github-actions"
ssh-keyscan -H ssh.railway.com >> ~/.ssh/known_hosts   # if you have not already
```

Test once:

```bash
railway ssh -i ~/.ssh/railway_github_actions -s web -- python /app/scripts/daily_update.py --dry-run
```

### 5. Collect Railway IDs

```bash
railway status
```

Or from the Railway dashboard URL / project settings:

| Value | Example | GitHub secret name |
|-------|---------|-------------------|
| Project ID | UUID | `RAILWAY_PROJECT_ID` |
| Service name | `web` | `RAILWAY_SERVICE` |
| Environment | `production` | `RAILWAY_ENVIRONMENT` |

### 6. Add GitHub repository secrets

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret | Value |
|--------|--------|
| `RAILWAY_SSH_PRIVATE_KEY` | Full contents of `~/.ssh/railway_github_actions` (private key) |
| `RAILWAY_PROJECT_ID` | Project UUID from step 5 |
| `RAILWAY_SERVICE` | `web` |
| `RAILWAY_ENVIRONMENT` | `production` (or your env name) |
| `RAILWAY_API_URL` | Optional — `https://<your-railway-domain>` for post-run smoke check |

### 7. Enable Actions and test

1. GitHub → **Actions** → workflow **Daily update (Railway)**  
2. **Run workflow** → **Run workflow** (manual)  
3. Wait ~5–20 minutes (news collection is the slow step)  
4. Confirm green check, then:

```bash
curl -s https://<your-railway-domain>/api/data-status | python3 -m json.tool
```

`latest_price_date` and `primary_prediction_date` should match the latest session.

---

## What runs each weekday

Workflow: [`.github/workflows/daily-update.yml`](../.github/workflows/daily-update.yml)

```text
GitHub Actions (22:00 UTC Mon–Fri)
    → railway ssh into service web
    → python scripts/daily_update.py
    → python scripts/stamp_published.py
```

Steps inside `daily_update.py`: collect prices/news → labels → score models → ensemble → evaluate.

---

## Manual catch-up (still useful)

| Situation | Command |
|-----------|---------|
| Missed a few days | Re-run the GitHub Action manually, or SSH: `railway ssh -s web -- python /app/scripts/daily_update.py` |
| Long downtime (>60d) | SSH with `--lookback-days 90` |
| Retrain models | Local `run_pipeline.py`, then `publish_deploy_bundle.py` (models change) |

---

## Laptop cron (optional backup)

`scripts/local_cron.sh` + `docs/local_cron.plist.example` still work when the Mac is awake. You can disable launchd once GitHub Actions is reliable:

```bash
launchctl bootout gui/$(id -u)/com.news-to-alpha.daily 2>/dev/null || true
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| SSH permission denied in Actions | Re-run step 4; verify `RAILWAY_SSH_PRIVATE_KEY` includes the full private key (with newlines) |
| `ModuleNotFoundError: yfinance` | Railway image not rebuilt — trigger redeploy on `main` |
| `NEWS_API_KEY` missing | Add variable on Railway `web` service |
| Job times out at 45 min | Rare; re-run; check Finnhub rate limits in logs |
| `is_current: false` after success | Prices may not have today’s bar yet — run after ~4:15 PM ET |

---

## Related docs

- [DATA.md § Daily refresh](DATA.md)
- [DEPLOY.md](DEPLOY.md) (operator deployment)
- [LOCAL_TESTING.md](LOCAL_TESTING.md)
