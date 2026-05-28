# Documentation

| Doc | Audience | Contents |
|-----|----------|----------|
| [../README.md](../README.md) | Everyone | Quick start, architecture, presets |
| [DATA.md](DATA.md) | Everyone | **Data pipeline** — artifacts, daily update, cron, publish to Railway |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Contributors | Module-by-module architecture (no runbooks) |
| [../web/README.md](../web/README.md) | Frontend | Next.js local dev + Vercel deploy |

**Local-only** (gitignored — machine-specific or private):

| Doc | Contents |
|-----|----------|
| [DEPLOY.md](DEPLOY.md) | One-time Railway + Vercel setup (generic URLs) |
| [DEPLOY_UI.md](DEPLOY_UI.md) | Your live URLs + smoke-test shortcuts |
| [PERSONAL_FULL_GUIDE.md](PERSONAL_FULL_GUIDE.md) | UI quirks, experiment notes, known issues |
| [local_cron.plist.example](local_cron.plist.example) | macOS launchd template |

**Where to look:**

- Update predictions → [DATA.md § Daily refresh](DATA.md)
- Mac asleep / missed cron → [DATA.md § Scheduled updates](DATA.md)
- First deploy → [DEPLOY.md](DEPLOY.md) (local) or ask for [DEPLOY_UI.md](DEPLOY_UI.md) cheat sheet
- How a module works → [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
