# Documentation

Reading order for contributors:

1. **[../README.md](../README.md)** — overview, quick start, results snapshot  
2. **[DEVELOPMENT.md](DEVELOPMENT.md)** — local setup, API/UI, smoke tests  
3. **[DATA.md](DATA.md)** — pipeline, artifacts, daily refresh, publishing  
4. **[RESULTS.md](RESULTS.md)** — evaluation metrics and model comparison  
5. **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** — architecture and design decisions  
6. **[../web/README.md](../web/README.md)** — frontend structure and deployment  

---

## Reference

| Doc | Contents |
|-----|----------|
| [DEVELOPMENT.md](DEVELOPMENT.md) | Environment setup, run locally, tests |
| [DATA.md](DATA.md) | Training presets, daily inference, artifact layout |
| [RESULTS.md](RESULTS.md) | Held-out test metrics, subsets, findings |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Modules, data flow, API surface |

---

## Common tasks

| Task | Where |
|------|-------|
| First-time train | [DATA.md § Full retrain](DATA.md) |
| Daily prediction refresh | [DATA.md § Daily refresh](DATA.md) |
| Publish inference bundle | [DATA.md § Publish](DATA.md) |
| Run API + UI locally | [DEVELOPMENT.md](DEVELOPMENT.md) |
| Interpret accuracy numbers | [RESULTS.md](RESULTS.md) |
| Understand a module | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
