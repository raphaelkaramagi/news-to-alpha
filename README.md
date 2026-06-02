# Stock Price and Sentiment Predictor

Binary classifier for **next-session stock direction** (UP/DOWN). Combines price history (LSTM), news text (TF-IDF + FinBERT embeddings), and a learned ensemble meta-model.

**Universe:** 20 US equities — AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR, GOOGL, MSFT, MU, AMD, AMZN

**Live demo:** [stock.raphaelkaramagi.com](https://news-to-alpha.vercel.app/) (Next.js on Vercel, Flask API on Railway)

---

## Results at a glance

Held-out **LSTM test split** (n=2,220, Dec 2025–Jun 2026). For honest news/ensemble comparison use the **`news_scored`** subset (n=399) where news models are out-of-sample. Full tables: **[docs/RESULTS.md](docs/RESULTS.md)**.

| Model | Test accuracy (all) | `news_scored` accuracy | `news_scored` AUC | Notes |
|-------|---------------------|------------------------|-------------------|-------|
| **Ensemble** | 50.6% | 50.1% | 0.492 | Conditional HGB meta-model |
| Previous-day direction | 51.5% | 52.6% | 0.525 | Simple baseline |
| LSTM (price) | 50.7% | 51.1% | 0.504 | Val AUC 0.528; test ≈ noise |
| TF-IDF news | 48.7% | 48.1% | 0.496 | Uncalibrated (sigmoid rejected) |
| FinBERT embeddings | 49.0% | 49.6% | 0.487 | Per-headline mean-pool |

Earlier reported ~62% ensemble accuracy was inflated by a **train→test news leakage** in the eval join (fixed June 2026). The live site was restored with a schema-compatible 13-feature ensemble; improved republish waits until honest `news_scored` accuracy ≥ **55%**.

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
| [docs/RESULTS.md](docs/RESULTS.md) | Leakage-free evaluation metrics |
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Architecture, splits, mermaid diagrams |
| [web/README.md](web/README.md) | Frontend |

After market close, refresh predictions without retraining:

```bash
python scripts/daily_update.py
```

Republish model artifacts only after a local retrain:

```bash
python scripts/publish_deploy_bundle.py --target railway --service web
```

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

## Testing

```bash
pytest tests/unit -q
cd web && npm run build
python scripts/publish_deploy_bundle.py --dry-run
```

See **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** for the full verification checklist.
