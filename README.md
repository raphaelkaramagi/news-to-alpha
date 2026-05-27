# News-to-Alpha

Predict whether a stock will go **up or down** the next trading session by combining price (LSTM), news (TF-IDF + embeddings), and a learned ensemble meta-model.

**15 tickers:** AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR

---

## Architecture

| Component | Stack | Role |
|-----------|-------|------|
| **UI** | Next.js 16 (`web/`) | Markets grid, ticker detail, status — deployed on Vercel |
| **API** | Flask (`app/server.py`) | JSON endpoints — deployed on Railway |
| **ML** | PyTorch + scikit-learn | Trained locally; only artifacts uploaded to cloud |

```
Mac: train + daily_update  →  publish bundle  →  Railway /data
Vercel (web/)  ──proxy──→  Flask API
```

---

## Quick start (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add NEWS_API_KEY

python scripts/setup_database.py
python scripts/run_pipeline.py --preset balanced
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

After market close, advance dates without retraining:

```bash
python scripts/daily_update.py
python scripts/publish_deploy_bundle.py --target railway
```

Full docs: **[docs/DATA.md](docs/DATA.md)** (artifacts, scripts, freshness, confidence).

Deploy: **[docs/DEPLOY_UI.md](docs/DEPLOY_UI.md)** (Vercel + Railway).

Doc index: **[docs/README.md](docs/README.md)** (public vs gitignored personal docs).

---

## UI pages

| Route | Description |
|-------|-------------|
| `/` | Markets grid, outcome dots, all-ticker price/accuracy overview (7d/30d/90d) |
| `/t/[symbol]` | Call, headlines, Why this call, Advanced, synced charts |
| `/status` | Data freshness |

Global date picker syncs all pages. **Restart Flask** after API changes (`python app/server.py --port 8000`).

Personal references (gitignored, local only): `docs/PERSONAL_FULL_GUIDE.md`, `docs/DEPLOY_VERCEL.md`

---

## Project layout

```
app/server.py           Flask JSON API
web/                    Next.js UI (Vercel) — npm commands run here
scripts/
  run_pipeline.py       Full train orchestrator
  daily_update.py       Collect + infer + ensemble (no retrain)
  score_models.py       Live LSTM scoring
  publish_deploy_bundle.py   Trim + upload to Railway
src/
  ml/lstm_live_export.py     Score dates after last label
  ml/ensemble_explain.py     Why-this-call counterfactuals
  features/sequence_generator.py
  utils/pipeline_config.py
  utils/trading_calendar.py
data/                   Local only (gitignored) — DB, CSVs, models
tests/unit/             pytest suite
docs/
  README.md             Doc index (public vs private)
  DATA.md               How to update data
  DEPLOY_UI.md          Vercel + Railway deploy
  DEPLOY.md             Railway API details
  PROJECT_OVERVIEW.md   Deep architecture reference
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
python scripts/run_pipeline.py --preset balanced   # 5 tickers, 3 seeds
python scripts/run_pipeline.py --preset advanced   # full universe + FinBERT
```

All scripts accept `--help`.
