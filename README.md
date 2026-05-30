# Stock Price and Sentiment Predictor

Binary classifier for **next-session stock direction** (UP/DOWN). Combines price history (LSTM), news text (TF-IDF + FinBERT embeddings), and a learned ensemble meta-model.

**Universe:** 20 US equities — AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR, GOOGL, MSFT, MU, AMD, AMZN

**Live demo:** [stock.raphaelkaramagi.com](https://stock.raphaelkaramagi.com) (Next.js on Vercel, Flask API on Railway)

---

## Results at a glance

Held-out **test split**, `max_v2` preset (chronological 70/15/15, next-day horizon, conditional ensemble). Full tables: **[docs/RESULTS.md](docs/RESULTS.md)**.

| Model | Test accuracy | AUC | Notes |
|-------|---------------|-----|-------|
| **Ensemble** | **62.2%** | **0.661** | Best overall; combines weak base models |
| Ensemble (high confidence) | **65.9%** | 0.676 | Confidence ≥ 0.25 |
| FinBERT embeddings | 53.9% | 0.603 | Strongest standalone news signal |
| TF-IDF | 50.9% | 0.640 | High AUC, very low UP recall |
| LSTM (price) | 49.7% | 0.495 | Near random; known collapse on test |
| Previous-day direction | 50.9% | 0.509 | Simple baseline |

News-driven models carry most usable signal; the ensemble learns to weight them on headline days. The LSTM price head remains a weak contributor.

---

## Architecture

```
Training host          →  collect, train, daily inference
         ↓ publish_deploy_bundle.py (SSH)
Railway /data volume   ←  Flask API (read-only, Dockerfile.inference)
         ↑
Vercel (web/)          ←  Next.js dashboard proxies /api/*
```

| Layer | Stack | Role |
|-------|-------|------|
| **UI** | Next.js 16 (`web/`) | Markets grid, ticker detail, status |
| **API** | Flask (`app/server.py`) | Serves CSVs + SQLite from `/data` |
| **ML** | PyTorch + scikit-learn | Trained offline; artifacts uploaded to the API host |

Training and raw data stay on the operator host. Production serves precomputed predictions only.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set NEWS_API_KEY (Finnhub)

python scripts/setup_database.py
python scripts/run_pipeline.py --preset max_v2
```

**API** (terminal 1):

```bash
python app/server.py --port 8000
```

**UI** (terminal 2, from `web/`):

```bash
cd web && cp .env.example .env.local && npm install && npm run dev
```

Open http://localhost:3000

Full setup and smoke tests: **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/README.md](docs/README.md) | Index and reading order |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local setup, API/UI, tests |
| [docs/DATA.md](docs/DATA.md) | Pipeline, artifacts, daily refresh, publish |
| [docs/RESULTS.md](docs/RESULTS.md) | Evaluation metrics |
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Architecture and modules |
| [web/README.md](web/README.md) | Frontend |

After market close, refresh predictions without retraining:

```bash
python scripts/daily_update.py
python scripts/publish_deploy_bundle.py --target railway --service web
```

Weekday automation is available via `.github/workflows/daily-update.yml` (see [DATA.md § Scheduled updates](DATA.md)).

---

## UI routes

| Route | Description |
|-------|-------------|
| `/` | Markets grid, resolved calls, overview chart |
| `/t/[symbol]` | Close-to-close hero, headlines, Why tab, charts |
| `/status` | Data freshness and evaluation summary |

A global date picker syncs all pages.

---

## Project layout

```
app/           Flask JSON API
web/           Next.js UI
scripts/       CLI pipeline (train, collect, publish)
src/           ML modules (features, models, data collection)
tests/unit/    pytest suite
docs/          Reference documentation
data/          Artifacts (gitignored — created by pipeline)
```

---

## Presets

```bash
python scripts/run_pipeline.py --preset quick      # 2 tickers, smoke test
python scripts/run_pipeline.py --preset balanced   # 5 tickers, 3-day horizon
python scripts/run_pipeline.py --preset advanced   # all tickers, 3-day + FinBERT
python scripts/run_pipeline.py --preset max        # all tickers, next-day (legacy)
python scripts/run_pipeline.py --preset max_v2     # recommended
```

All scripts accept `--help`.

---

## Testing

```bash
pytest tests/unit -q
cd web && npm run build
python scripts/publish_deploy_bundle.py --dry-run
```

See **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** for the full verification checklist.
