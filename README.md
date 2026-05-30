# Stock Price and Sentiment Predictor

Binary classifier for **next-session stock direction** (UP/DOWN). Combines price history (LSTM), news text (TF-IDF + FinBERT embeddings), and a learned ensemble meta-model.

**Tickers:** 20 US equities — AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR, GOOGL, MSFT, MU, AMD, AMZN

**Demo:** [stock.raphaelkaramagi.com](https://stock.raphaelkaramagi.com) (example deployment: Next.js on Vercel → Flask API on Railway)

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

**Takeaway:** News-driven models carry most usable signal; the ensemble learns to weight them on headline days. The LSTM price head remains a weak contributor and is an active improvement area.

---

## Architecture

```
Training host (local)  →  collect, train, daily inference
         ↓ publish_deploy_bundle.py (SSH)
Railway /data volume   ←  Flask API (read-only, Dockerfile.inference)
         ↑
Vercel (web/)          ←  Next.js dashboard proxies /api/*
```

| Layer | Stack | Role |
|-------|-------|------|
| **UI** | Next.js 16 (`web/`) | Markets grid, ticker detail, status |
| **API** | Flask (`app/server.py`) | Serves CSVs + SQLite from `/data` |
| **ML** | PyTorch + scikit-learn | Trained offline; artifacts uploaded to API host |

Training and raw data stay on the operator machine. Production serves precomputed predictions only.

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

**Full local testing checklist:** [docs/LOCAL_TESTING.md](docs/LOCAL_TESTING.md)

---

## Documentation

| Read first | Doc | For |
|------------|-----|-----|
| 1 | [docs/README.md](docs/README.md) | Index and reading order |
| 2 | [docs/LOCAL_TESTING.md](docs/LOCAL_TESTING.md) | **Run API + UI, smoke tests, verify before deploy** |
| 3 | [docs/DATA.md](docs/DATA.md) | Pipeline, artifacts, daily refresh, publish |
| 4 | [docs/RESULTS.md](docs/RESULTS.md) | Evaluation metrics and findings |
| 5 | [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Architecture and design decisions |
| 6 | [web/README.md](web/README.md) | Frontend development |

After market close, refresh predictions without retraining:

```bash
python scripts/daily_update.py
python scripts/publish_deploy_bundle.py --target railway --service web   # local → Railway upload
```

**Automated (no laptop):** [docs/AUTOMATION.md](docs/AUTOMATION.md) — GitHub Actions runs `daily_update.py` on Railway Mon–Fri ~6 PM ET.

---

## UI routes

| Route | Description |
|-------|-------------|
| `/` | Markets grid, ✓/✗ resolved calls, overview chart |
| `/t/[symbol]` | Close-to-close hero, headlines, Why this call (gauge + drivers), charts |
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
docs/          Public reference documentation
data/          Artifacts (gitignored — created by pipeline)
```

---

## Presets

```bash
python scripts/run_pipeline.py --preset quick      # 2 tickers, smoke test
python scripts/run_pipeline.py --preset balanced   # 5 tickers, 3-day horizon
python scripts/run_pipeline.py --preset advanced   # all tickers, 3-day + FinBERT
python scripts/run_pipeline.py --preset max        # all tickers, next-day (legacy prod)
python scripts/run_pipeline.py --preset max_v2     # recommended: FinBERT + conditional ensemble
```

All scripts accept `--help`.

---

## Testing

```bash
pytest tests/unit -q
cd web && npm run build
python scripts/publish_deploy_bundle.py --dry-run
```

Step-by-step local verification (API, UI, smoke curls, checklist): **[docs/LOCAL_TESTING.md](docs/LOCAL_TESTING.md)**.
