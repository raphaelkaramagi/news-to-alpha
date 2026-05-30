# Automated daily updates (GitHub Actions → Railway)

Run `daily_update.py` on the Railway `/data` volume **without** your laptop. No `publish_deploy_bundle.py` step — the job writes CSVs and SQLite where the API already reads them.

**Schedule:** Mon–Fri **22:00 UTC** (~6:00 PM ET in EDT, ~5 PM in EST). Configured in [`.github/workflows/daily-update.yml`](../.github/workflows/daily-update.yml) under `on.schedule.cron`. In EST/winter, change to `0 23 * * 1-5`.  
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
| **`RAILWAY_API_TOKEN`** | **Required for CI.** Railway → click your avatar → **Account Settings** → **Tokens** → **Create Token** → choose **No workspace** (not a specific workspace). Copy once. |
| `RAILWAY_SSH_PRIVATE_KEY` | Full contents of `~/.ssh/railway_github_actions` (private key) |
| `RAILWAY_PROJECT_ID` | Project UUID from step 5 |
| `RAILWAY_SERVICE` | `web` |
| `RAILWAY_ENVIRONMENT` | `production` (or your env name) |
| `RAILWAY_API_URL` | Optional — `https://<your-railway-domain>` for post-run smoke check |

> **Why `RAILWAY_API_TOKEN`?** `railway ssh` in GitHub Actions needs an **account** token (`RAILWAY_API_TOKEN`). Project tokens (`RAILWAY_TOKEN`) and SSH keys alone are not enough — that causes `Unauthorized. Please login with railway login`.

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

## What happened if a run fails after ~30–40 minutes?

The old path **re-scored all ~2,400 FinBERT rows** on Railway CPU (~3 GB RAM). That often ends with **exit code 1** and no Python traceback — usually **OOM (out of memory)** or SSH disconnect, not a logic bug.

**Your site is fine** if you already published recently: the failed run stopped before `build_ensemble` / `final_ensemble_predictions.csv`. Partial TF-IDF/LSTM writes on `/data` do not affect the live CSV the API serves until ensemble completes.

**Do not re-run the old workflow** until Railway has redeployed from `main` (includes `--incremental` scoring). Then manual runs take **~5–15 minutes** for all 20 tickers.

---

## After a failed run — checklist

1. **Cancel** the workflow if still running (GitHub → Actions → Cancel).
2. **Confirm Railway redeployed** from latest `main` (Dashboard → web → Deployments → success).
3. **Optional but recommended once:** from your laptop, sync full artifacts to Railway:
   ```bash
   python scripts/daily_update.py
   python scripts/publish_deploy_bundle.py --target railway --service web
   ```
   This refreshes `pipeline_config.json` (20 tickers, horizon 1) and prediction CSVs on `/data`.
4. **Re-run** Actions → Daily update (Railway) → Run workflow manually.
5. Expect logs like:
   ```text
   tickers       : [all 20 …]
   horizon       : 1
   [score_embeddings] Incremental – skipping full FinBERT historical rescore
   ```

---

Workflow: [`.github/workflows/daily-update.yml`](../.github/workflows/daily-update.yml)

```text
GitHub Actions (22:00 UTC Mon–Fri)
    → railway ssh into service web
    → python scripts/daily_update.py
    → python scripts/stamp_published.py
```

Steps inside `daily_update.py`: collect prices/news → labels → score models (**incremental** — live rows only) → ensemble → evaluate.

**Typical runtime:** 3–10 minutes after the incremental scoring fix. The first run (or a full rescore without `--incremental`) can take **30–90+ minutes** on Railway CPU because FinBERT re-encodes every historical headline.

### Cost notes

Railway bills **per minute of RAM + vCPU used**. A daily FinBERT burst (~3 GB RAM, high CPU for 30–60 min) costs roughly **$0.05–0.15 per run** — not hundreds of dollars. The **$166 estimated bill** in the dashboard extrapolates a short spike as if it ran 24/7 all month; ignore it unless usage stays elevated continuously.

With incremental scoring, weekday runs should stay well within the Hobby **$5 included usage**.

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
| `Unauthorized. Please login with railway login` | Add **`RAILWAY_API_TOKEN`** (account token, **No workspace**). Do not use a project token here |
| `ModuleNotFoundError: yfinance` | Railway image not rebuilt — trigger redeploy on `main` |
| `NEWS_API_KEY` missing | Add variable on Railway `web` service |
| Job times out at 45 min | First run may full-rescore FinBERT — wait for deploy with `--incremental`, then re-run |
| Run stuck at `Loading weights` / FinBERT | Usually **not stuck** — full historical embedding rescore with no progress bar; can take 30–90 min |
| `is_current: false` after success | Prices may not have today’s bar yet — run after ~4:15 PM ET |

---

## Related docs

- [DATA.md § Daily refresh](DATA.md)
- [DEPLOY.md](DEPLOY.md) (operator deployment)
- [LOCAL_TESTING.md](LOCAL_TESTING.md)
