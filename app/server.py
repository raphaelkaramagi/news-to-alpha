#!/usr/bin/env python3
"""
News-to-Alpha web app.

Search a ticker → see the latest prediction, confidence score, and
the headlines that drove it.

Usage:
    python app/server.py
    python app/server.py --port 5001
"""

import json
import logging
import sqlite3
import argparse
import threading
from datetime import date
from pathlib import Path

from flask import Flask, render_template, request, jsonify, redirect, url_for

# ── Paths ────────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"

import sys
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from src.config import DATABASE_PATH, TICKERS, TICKER_TO_COMPANY, MODELS_DIR, PROCESSED_DATA_DIR
except ModuleNotFoundError:
    DATABASE_PATH = _PROJECT_ROOT / "data" / "database.db"
    TICKERS = [
        "AAPL", "NVDA", "WMT", "LLY", "JPM",
        "XOM", "MCD", "TSLA", "DAL", "MAR",
        "GS", "NFLX", "META", "ORCL", "PLTR",
    ]
    TICKER_TO_COMPANY = {}
    MODELS_DIR = _PROJECT_ROOT / "data" / "models"
    PROCESSED_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

app = Flask(__name__, template_folder=str(_APP_DIR / "templates"))

# ── Refresh state ────────────────────────────────────────────────────────────
_refresh_lock = threading.Lock()
_refresh_status: dict = {"running": False, "last_date": None, "error": None, "step": None}


def _run_tfidf_training() -> dict:
    """Run the TF-IDF baseline pipeline and upsert predictions into the DB."""
    import train_nlp as tn

    db_path = Path(DATABASE_PATH)
    df = tn.build_dataset(db_path)
    train_df, val_df, test_df = tn.chronological_split(df)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    pipe = tn.train(train_df)

    metrics = {}
    for name, split_df in splits.items():
        metrics[name] = tn.evaluate(pipe, split_df, name)

    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / tn.MODEL_FILE)
    tn.save_predictions_csv(pipe, splits, PROCESSED_DATA_DIR / "news_tfidf_predictions.csv")
    tn.upsert_predictions_db(pipe, splits, db_path)
    return {"model": "news_tfidf", "metrics": metrics}


def _run_embeddings_training() -> dict:
    """Run the sentence-embeddings pipeline and upsert predictions into the DB."""
    import train_nlp as tn
    import train_news_embeddings as tne

    tne._ensure_hf_home_for_downloads()
    db_path = Path(DATABASE_PATH)
    df = tn.build_dataset(db_path)
    train_df, val_df, test_df = tn.chronological_split(df)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=1_000, class_weight="balanced",
                             random_state=42, solver="lbfgs")
    model = tne.NewsEmbeddingClassifier(
        sentence_model_name=tne.DEFAULT_SENTENCE_MODEL, classifier=clf)
    model.fit(train_df["headlines_text"], train_df["label_binary"])

    metrics = {}
    for name, split_df in splits.items():
        metrics[name] = tne.evaluate(model, split_df, name)

    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": model.classifier,
                 "sentence_model_name": model.sentence_model_name,
                 "batch_size": model.batch_size,
                 "model_name": tne.MODEL_NAME,
                 "model_version": tne.MODEL_VERSION},
                MODELS_DIR / tne.MODEL_FILE)
    tne.save_predictions_csv(model, splits, PROCESSED_DATA_DIR / tne.CSV_NAME)
    tne.upsert_predictions_db(model, splits, db_path)
    return {"model": "news_embeddings", "metrics": metrics}


def _collect_fresh_data() -> dict:
    """Fetch latest prices, news, and generate labels."""
    from datetime import datetime, timedelta
    from src.database.schema import DatabaseSchema
    from src.data_collection.price_collector import PriceCollector
    from src.data_collection.news_collector import NewsCollector
    from src.data_processing.label_generator import LabelGenerator

    log = logging.getLogger(__name__)
    DatabaseSchema().create_all_tables()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    summary = {}

    log.info("Collecting prices %s → %s …", start_date, end_date)
    _refresh_status["step"] = "Collecting prices…"
    price_stats = PriceCollector().collect(TICKERS, start_date, end_date)
    summary["prices"] = price_stats["rows_added"]

    try:
        from src.config import FINNHUB_API_KEY
    except ImportError:
        FINNHUB_API_KEY = ""

    if FINNHUB_API_KEY:
        log.info("Collecting news %s → %s …", start_date, end_date)
        _refresh_status["step"] = "Collecting news…"
        news_stats = NewsCollector(api_key=FINNHUB_API_KEY).collect(
            TICKERS, start_date, end_date)
        summary["news"] = news_stats["rows_added"]
    else:
        log.warning("No FINNHUB_API_KEY — skipping news collection")
        summary["news"] = 0

    log.info("Generating labels …")
    _refresh_status["step"] = "Generating labels…"
    label_summary = LabelGenerator().generate(TICKERS)
    summary["labels"] = label_summary["total_labels"]

    return summary


def _refresh_predictions() -> None:
    """Collect fresh data, then retrain all models (background thread target)."""
    global _refresh_status
    log = logging.getLogger(__name__)
    results = []

    try:
        data_summary = _collect_fresh_data()
        log.info("Data collected: %s", data_summary)
    except Exception as exc:
        log.warning("Data collection failed: %s", exc)
        results.append({"model": "data_collection", "error": str(exc)})

    _refresh_status["step"] = "Training TF-IDF model…"
    try:
        results.append(_run_tfidf_training())
    except Exception as exc:
        log.warning("TF-IDF training failed: %s", exc)
        results.append({"model": "news_tfidf", "error": str(exc)})

    _refresh_status["step"] = "Training embeddings model…"
    try:
        _run_embeddings_training()
        results.append({"model": "news_embeddings"})
    except Exception as exc:
        log.warning("Embeddings training skipped: %s", exc)
        results.append({"model": "news_embeddings", "error": str(exc)})

    with _refresh_lock:
        _refresh_status["running"] = False
        _refresh_status["step"] = None
        _refresh_status["last_date"] = str(date.today())
        errors = [r["error"] for r in results if "error" in r]
        _refresh_status["error"] = "; ".join(errors) if errors else None


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _latest_prediction(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """Most recent prediction row for a ticker (across any model_version)."""
    row = conn.execute(
        """SELECT ticker, date, news_pred_proba, news_confidence,
                  news_pred_binary, news_top_headlines, actual_binary,
                  model_version
           FROM predictions
           WHERE ticker = ?
           ORDER BY date DESC, created_at DESC
           LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["headlines_list"] = json.loads(d["news_top_headlines"] or "[]")
    except (json.JSONDecodeError, TypeError):
        raw = d.get("news_top_headlines") or ""
        d["headlines_list"] = [h.strip() for h in raw.split("|") if h.strip()]
    return d


def _recent_news(conn: sqlite3.Connection, ticker: str, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """SELECT title, source, published_at
           FROM news
           WHERE ticker = ? AND title IS NOT NULL AND title != ''
           ORDER BY published_at DESC
           LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _recent_prices(conn: sqlite3.Connection, ticker: str, limit: int = 5) -> list[dict]:
    rows = conn.execute(
        """SELECT date, open, high, low, close, volume
           FROM prices
           WHERE ticker = ?
           ORDER BY date DESC
           LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _latest_label(conn: sqlite3.Connection, ticker: str) -> dict | None:
    row = conn.execute(
        """SELECT date, label_binary, label_return, close_t, close_t_plus_1
           FROM labels
           WHERE ticker = ?
           ORDER BY date DESC
           LIMIT 1""",
        (ticker,),
    ).fetchone()
    return dict(row) if row else None


def _ticker_summary(conn: sqlite3.Connection, ticker: str) -> dict:
    """Aggregate everything the app shows for a single ticker."""
    prediction = _latest_prediction(conn, ticker)
    news = _recent_news(conn, ticker)
    prices = _recent_prices(conn, ticker)
    label = _latest_label(conn, ticker)

    if prediction and not prediction["headlines_list"] and news:
        prediction["headlines_list"] = [a["title"] for a in news[:5]]

    return {
        "ticker": ticker,
        "company": TICKER_TO_COMPANY.get(ticker, ticker),
        "prediction": prediction,
        "recent_news": news,
        "recent_prices": prices,
        "latest_label": label,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    q = request.args.get("q", "").strip().upper()
    conn = _get_db()
    try:
        selected = None
        if q and q in TICKERS:
            selected = _ticker_summary(conn, q)
        return render_template(
            "index.html",
            tickers=TICKERS,
            ticker_to_company=TICKER_TO_COMPANY,
            query=q,
            selected=selected,
            refresh_status=_refresh_status,
            today=str(date.today()),
        )
    finally:
        conn.close()


@app.route("/refresh", methods=["POST"])
def refresh():
    """Kick off model retraining in a background thread (once per day)."""
    global _refresh_status
    with _refresh_lock:
        if _refresh_status["running"]:
            return redirect(request.referrer or url_for("index"))
        if _refresh_status["last_date"] == str(date.today()):
            return redirect(request.referrer or url_for("index"))
        _refresh_status = {"running": True, "last_date": None, "error": None, "step": "Starting…"}

    thread = threading.Thread(target=_refresh_predictions, daemon=True)
    thread.start()
    return redirect(request.referrer or url_for("index"))


@app.route("/api/refresh-status")
def api_refresh_status():
    return jsonify(_refresh_status)


@app.route("/api/ticker/<ticker>")
def api_ticker(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
    conn = _get_db()
    try:
        return jsonify(_ticker_summary(conn, ticker))
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News-to-Alpha web app")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  News-to-Alpha app running at http://localhost:{args.port}")
    print(f"  Database: {DATABASE_PATH}\n")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
