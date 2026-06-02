# Stock Price and Sentiment Predictor

Next-session **direction** (UP/DOWN) and **expected move** (±% band) for 20 US equities. Combines price (LSTM), news (TF-IDF + FinBERT), a volatility regressor, and a conditional ensemble meta-model.

**Universe:** AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR, GOOGL, MSFT, MU, AMD, AMZN

**Live demo:** [news-to-alpha.vercel.app](https://news-to-alpha.vercel.app/) (Next.js on Vercel, Flask API on Railway)

---

## Signal limits 

| Output | Test skill | Notes |
|--------|------------|-------|
| **Direction** (UP/DOWN) | AUC ≈ **0.50** | Efficient-market ceiling for liquid large-cap daily direction |
| **Expected move %** (volatlity/return) | high-move AUC ≈ **0.65**, MAE ≈ 1.2% | The measurable signal; shown as ±% band on cards and detail |
| **Drift** (always UP) | 52.5% (1d) | Base rate, not skill — accuracy without AUC does not count |

Full metrics and subsets: **[docs/RESULTS.md](docs/RESULTS.md)** 

---

## Architecture

```
Training host  →  collect, train
       ↓ publish_deploy_bundle.py
Railway /data  ←  Flask API 
       ↑
Vercel (web/)  ←  Next.js dashboard
```

| Layer | Stack | Role |
|-------|-------|------|
| **UI** | Next.js 16 (`web/`) | Markets grid, direction call, expected-move band |
| **API** | Flask (`app/server.py`) | Serves CSVs + SQLite from `/data` |
| **ML** | PyTorch + scikit-learn | Trained offline; artifacts uploaded to API host |

Models: **LSTM** (price direction), **TF-IDF + FinBERT** (news direction), **volatility regressor** (next-day |return|), **conditional HGB ensemble** (13 features, routes has_news / no_news).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set NEWS_API_KEY (Finnhub)

python scripts/setup_database.py
python scripts/run_pipeline.py --preset max_v2
```

**API:** `python app/server.py --port 8000`  
**UI:** `cd web && npm install && npm run dev` → http://localhost:3000

Details: **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/README.md](docs/README.md) | Index |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Local setup, smoke tests |
| [docs/DATA.md](docs/DATA.md) | Pipeline, artifacts, publish |
| [docs/RESULTS.md](docs/RESULTS.md) | Evaluation metrics |
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Architecture |
| [web/README.md](web/README.md) | Frontend |

Daily refresh (no retrain): `python scripts/daily_update.py`  
Republish artifacts: `python scripts/publish_deploy_bundle.py --target railway --service web`

---

## UI routes

| Route | Description |
|-------|-------------|
| `/` | Markets grid — direction + expected move |
| `/t/[symbol]` | Hero band, headlines, Why / Advanced |
| `/status` | Freshness and evaluation summary |

---

## Testing

```bash
pytest tests/unit -q
cd web && npm run build
python scripts/publish_deploy_bundle.py --dry-run
```
