# Documentation index


| Doc | Audience | Contents |
|-----|----------|----------|
| [../README.md](../README.md) | Everyone | Quick start, architecture, layout |
| [DATA.md](DATA.md) | Developers | Artifacts, daily update, freshness, “Why this call” |
| [DEPLOY_UI.md](DEPLOY_UI.md) | Deploy | Full stack: Mac train → Railway API → Vercel UI |
| [DEPLOY.md](DEPLOY.md) | Deploy | Flask API only (Railway / Render / Fly) |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Deep dive | Module-by-module architecture reference |
| [../web/README.md](../web/README.md) | Frontend | Next.js local dev + Vercel settings |

## Private (gitignored — local only)

These files may contain personal URLs, cron paths, or internal notes. They live in `docs/` for convenience but are **not** pushed to GitHub:

| Doc | Purpose |
|-----|---------|
| `DEPLOY_VERCEL.md` | Step-by-step deploy with your Railway/Vercel URLs |
| `PERSONAL_FULL_GUIDE.md` | Single reference: architecture, UI behavior, troubleshooting |
| `local_cron.plist.example` | launchd template with paths to edit |
| `ROADMAP.md` | Internal planning |
| `session*_contract.md` | Early API contracts (historical) |
| `CONTEXT_FOR_NEW_CHAT.md` | Paste-in context for AI sessions |

Copy `local_cron.plist.example` to `~/Library/LaunchAgents/` and edit paths locally.

## Typical workflows

**Local dev**

```bash
# Terminal 1
source .venv/bin/activate && python app/server.py --port 8000

# Terminal 2
cd web && npm run dev
```

**Refresh predictions**

```bash
python scripts/daily_update.py
python scripts/publish_deploy_bundle.py --target railway
```

**After pulling API/UI changes:** restart Flask (port 8000). If “Address already in use”, kill the old process first (`lsof -nP -iTCP:8000 -sTCP:LISTEN`).
