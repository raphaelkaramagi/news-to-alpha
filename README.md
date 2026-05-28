# Stock Price and Sentiment Predictor

Predict whether a stock will go **up or down** the next trading session by combining price (LSTM), news (TF-IDF + embeddings), and a learned ensemble meta-model.

**Live site:** [stock.raphaelkaramagi.com](https://stock.raphaelkaramagi.com) (Vercel UI → Railway API)

**20 tickers:** AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR, GOOGL, MSFT, MU, AMD, AMZN

---

## Architecture

| Component | Stack | Role |
|-----------|-------|------|
| **UI** | Next.js 16 (`web/`) | Markets grid, ticker detail, status — Vercel |
| **API** | Flask (`app/server.py`) | JSON endpoints — Railway + `/data` volume |
| **ML** | PyTorch + scikit-learn | Trained on your Mac; only artifacts uploaded |

```
Mac: train + daily_update  →  publish bundle (SSH)  →  Railway /data
Vercel (web/)  ──proxy──→  Flask API
```

---

## Quick start (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add NEWS_API_KEY

python scripts/setup_database.py
python scripts/run_pipeline.py --preset max
```

**Terminal 1 — API**

```bash
python app/server.py --port 8000
```

**Terminal 2 — UI** (must run from `web/`)

```bash
cd web && cp .env.example .env.local && npm install && npm run dev
```

Open http://localhost:3000

---

## Updating predictions

After market close:

```bash
python scripts/daily_update.py
python scripts/publish_deploy_bundle.py --target railway --service web
```

Mac asleep at cron time? One manual run catches up — see [docs/DATA.md § Scheduled updates](docs/DATA.md). Full reference: **[docs/README.md](docs/README.md)**.

---

## UI pages

| Route | Description |
|-------|-------------|
| `/` | Markets grid, outcome dots, all-ticker overview |
| `/t/[symbol]` | Call, headlines, Why this call, Advanced, charts |
| `/status` | Data freshness |

Global date picker syncs all pages.

---

## Project layout

```
app/server.py           Flask JSON API
web/                    Next.js UI — npm commands run here
scripts/
  run_pipeline.py       Full train orchestrator
  daily_update.py       Collect + infer + ensemble (no retrain)
  audit_data_coverage.py   Price/news/label/live-row coverage report
  publish_deploy_bundle.py   Trim + SSH upload to Railway
src/                    ML pipeline modules
data/                   Local only (gitignored)
tests/unit/             pytest suite
docs/                   Public reference docs
```

---

## Testing

```bash
pytest tests/unit -q
cd web && npm run build
python scripts/publish_deploy_bundle.py --dry-run
```

---

## Presets

```bash
python scripts/run_pipeline.py --preset quick      # 2 tickers, fast
python scripts/run_pipeline.py --preset balanced   # 5 tickers, 3-day horizon
python scripts/run_pipeline.py --preset advanced   # all tickers, 3-day + FinBERT
python scripts/run_pipeline.py --preset max        # all tickers, next-day — production deploy
python scripts/run_pipeline.py --preset max_v2     # accuracy experiment (FinBERT + conditional ensemble)
```

All scripts accept `--help`.
