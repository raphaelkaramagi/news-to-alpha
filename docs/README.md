# Documentation

Reading order:

1. **[../README.md](../README.md)** — project summary, quick start, results snapshot  
2. **[LOCAL_TESTING.md](LOCAL_TESTING.md)** — run API + UI locally, smoke tests, pre-deploy checklist  
3. **[DATA.md](DATA.md)** — data flow, training presets, daily refresh, publishing  
4. **[RESULTS.md](RESULTS.md)** — evaluation metrics, model comparison, known limitations  
5. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** — architecture, modules, design decisions  
6. **[../web/README.md](../web/README.md)** — frontend setup and Vercel deployment  

---

## Public docs

| Doc | Contents |
|-----|----------|
| [LOCAL_TESTING.md](LOCAL_TESTING.md) | Local API/UI setup, daily_update checks, smoke tests |
| [DATA.md](DATA.md) | Artifacts, pipeline operations, scheduled updates, Railway publish |
| [RESULTS.md](RESULTS.md) | Test-set accuracy, AUC, subsets, key findings |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | System architecture and module reference (summary level) |

---

## Operator docs (gitignored)

These files are excluded from the public repository (see `.gitignore`). They contain deployment URLs, host-specific paths, and a complete file-by-file codebase map.

| Doc | Contents |
|-----|----------|
| [DEPLOY.md](DEPLOY.md) | Production deployment (Railway + Vercel), live URLs, costs |
| [PERSONAL_FULL_GUIDE.md](PERSONAL_FULL_GUIDE.md) | Full codebase inventory, operator runbook, troubleshooting |
| [local_cron.plist.example](local_cron.plist.example) | launchd template for scheduled daily updates |

---

## Common tasks

| Task | Where |
|------|-------|
| **Test locally (API + UI)** | [LOCAL_TESTING.md](LOCAL_TESTING.md) |
| First-time train | [DATA.md § Full retrain](DATA.md) |
| Daily prediction refresh | [DATA.md § Daily refresh](DATA.md) |
| Publish to production API | [DATA.md § Publish](DATA.md) |
| **Automated weekday updates (GitHub Actions)** | **[AUTOMATION.md](AUTOMATION.md)** |
| Scheduled weekday updates (Mac launchd) | [DATA.md § Scheduled updates](DATA.md) |
| Deploy UI + API | [DEPLOY.md](DEPLOY.md) (local) |
| Understand a module | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) or [PERSONAL_FULL_GUIDE.md](PERSONAL_FULL_GUIDE.md) |
| Interpret accuracy numbers | [RESULTS.md](RESULTS.md) |
