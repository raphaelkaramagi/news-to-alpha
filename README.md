# Stock Price and Sentiment Predictor

Next-session **direction** (UP/DOWN) and **expected move** (±% band) for 20 US equities. Combines price (LSTM), news (TF-IDF + FinBERT), a volatility regressor, and a conditional ensemble meta-model.

**Universe:** AAPL, NVDA, WMT, LLY, JPM, XOM, MCD, TSLA, DAL, MAR, GS, NFLX, META, ORCL, PLTR, GOOGL, MSFT, MU, AMD, AMZN

**Live demo:** [news-to-alpha.vercel.app](https://news-to-alpha.vercel.app/) (Next.js on Vercel, Flask API on Railway)

---

## Results at a glance

Held-out test split (n≈2,183), latest `max_v2` retrain.

Use **AUC**, not accuracy — daily direction sits near the efficient-market noise floor, so raw accuracy is mostly drift.

| Output | Test skill | Read as |
|--------|------------|---------|
| **Direction** (UP/DOWN, ensemble) | AUC ≈ **0.53** | The headline call; marginal, near noise floor |
| **Direction on news days** (`news_scored`, n≈380) | AUC ≈ **0.55** | Slight edge when fresh headlines exist |
| **Expected move %** (volatility) | high-move AUC ≈ **0.65**, MAE ≈ **1.2%** | More reliable than direction — shown as the ±% band |
| **Drift** (always UP) | acc ≈ 0.51 | Base rate, not skill |

Full metrics, subsets, and walk-forward findings: **[docs/RESULTS.md](docs/RESULTS.md)**

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

### Data inputs (what actually feeds the models)

| Source | Rows | Used by models? |
|--------|------|-----------------|
| Prices (yfinance, OHLCV + VIX/SPY) | ~16.5k | Yes — LSTM, volatility |
| News (Finnhub) + **FinBERT sentiment/relevance** | ~48k (all scored) | Yes — TF-IDF + embedding news models → ensemble |
| Fundamentals / macro (FRED) / earnings dates | 20 / 787 / 974 | **Collected but gated** — walk-forward showed no direction lift, so off by default (`--extra-features`) |
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

Setup, retrain, verify, and run locally, step by step: **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**

---

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/README.md](docs/README.md) | Index / reading order |
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Detailed overview: data, models, pipeline, modules, deployment |
| [docs/RESULTS.md](docs/RESULTS.md) | Evaluation metrics + walk-forward findings |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, retrain, verify, run API/UI locally |
| [docs/DATA.md](docs/DATA.md) | Pipeline, data sources, artifacts, publish |
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
