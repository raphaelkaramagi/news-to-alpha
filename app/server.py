#!/usr/bin/env python3
"""
News-to-Alpha web app.

Search a ticker → see the latest prediction, confidence score, and
the headlines that drove it.  Runs the full pipeline on refresh:
  data → NLP → features → split → LSTM → eval dataset → ensemble.

Usage:
    python app/server.py
    python app/server.py --port 5001
"""

import json
import logging
import sqlite3
import argparse
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
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

log = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(_APP_DIR / "templates"))

# ── Ensure database exists on startup ────────────────────────────────────────
try:
    from src.database.schema import DatabaseSchema
    DatabaseSchema().create_all_tables()
except Exception:
    pass

# ── Refresh state ────────────────────────────────────────────────────────────
_refresh_lock = threading.Lock()
_refresh_status: dict = {"running": False, "last_date": None, "error": None, "step": None}

# Ensemble CSV cached in memory after each refresh
_ensemble_cache: pd.DataFrame | None = None


def _load_ensemble_cache() -> pd.DataFrame | None:
    """Load the ensemble CSV into memory if it exists."""
    path = PROCESSED_DATA_DIR / "final_ensemble_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# Try loading at startup
_ensemble_cache = _load_ensemble_cache()


# ═════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEPS
# ═════════════════════════════════════════════════════════════════════════════

def _collect_fresh_data() -> dict:
    """Fetch latest prices (365 days for LSTM), news, and generate labels."""
    from src.database.schema import DatabaseSchema
    from src.data_collection.price_collector import PriceCollector
    from src.data_collection.news_collector import NewsCollector
    from src.data_processing.label_generator import LabelGenerator

    DatabaseSchema().create_all_tables()

    end_date = datetime.now().strftime("%Y-%m-%d")
    price_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    news_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    summary = {}

    log.info("Collecting prices %s → %s …", price_start, end_date)
    _refresh_status["step"] = "Collecting prices…"
    price_stats = PriceCollector().collect(TICKERS, price_start, end_date)
    summary["prices"] = price_stats["rows_added"]

    try:
        from src.config import FINNHUB_API_KEY
    except ImportError:
        FINNHUB_API_KEY = ""

    if FINNHUB_API_KEY:
        log.info("Collecting news %s → %s …", news_start, end_date)
        _refresh_status["step"] = "Collecting news…"
        news_stats = NewsCollector(api_key=FINNHUB_API_KEY).collect(
            TICKERS, news_start, end_date)
        summary["news"] = news_stats["rows_added"]
    else:
        log.warning("No FINNHUB_API_KEY — skipping news collection")
        summary["news"] = 0

    log.info("Generating labels …")
    _refresh_status["step"] = "Generating labels…"
    label_summary = LabelGenerator().generate(TICKERS)
    summary["labels"] = label_summary["total_labels"]

    return summary


def _nlp_splits(tn) -> tuple[pd.DataFrame, dict]:
    """Build NLP dataset and split using only news-bearing dates.

    The full dataset covers 365 days of labels, but Finnhub only returns ~21
    days of news.  If we split all dates chronologically, the training set has
    zero news and the models learn nothing.  Instead we:
      1. Split only on dates that have actual headlines (for train/val/test).
      2. Predict on ALL dates — placeholder rows get 0.5 which is correct.
    """
    db_path = Path(DATABASE_PATH)
    full_df = tn.build_dataset(db_path)

    has_news = full_df[full_df["headlines_text"] != tn.PLACEHOLDER_TEXT]

    if len(has_news) >= 10:
        train_df, val_df, test_df = tn.chronological_split(has_news)
        no_news = full_df[full_df["headlines_text"] == tn.PLACEHOLDER_TEXT]
        splits = {"train": train_df, "val": val_df, "test": test_df}
        all_splits = {"train": train_df, "val": val_df, "test": test_df,
                      "no_news": no_news}
    else:
        train_df, val_df, test_df = tn.chronological_split(full_df)
        splits = {"train": train_df, "val": val_df, "test": test_df}
        all_splits = splits

    return full_df, splits, all_splits


def _run_tfidf_training() -> dict:
    import train_nlp as tn

    db_path = Path(DATABASE_PATH)
    full_df, splits, all_splits = _nlp_splits(tn)

    pipe = tn.train(splits["train"])

    metrics = {}
    for name in ("train", "val", "test"):
        metrics[name] = tn.evaluate(pipe, splits[name], name)

    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / tn.MODEL_FILE)
    tn.save_predictions_csv(pipe, all_splits, PROCESSED_DATA_DIR / "news_tfidf_predictions.csv")
    tn.upsert_predictions_db(pipe, all_splits, db_path)
    return {"model": "news_tfidf", "metrics": metrics}


def _run_embeddings_training() -> dict:
    import train_nlp as tn
    import train_news_embeddings as tne

    tne._ensure_hf_home_for_downloads()
    db_path = Path(DATABASE_PATH)
    full_df, splits, all_splits = _nlp_splits(tn)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=1_000, class_weight="balanced",
                             random_state=42, solver="lbfgs")
    model = tne.NewsEmbeddingClassifier(
        sentence_model_name=tne.DEFAULT_SENTENCE_MODEL, classifier=clf)
    model.fit(splits["train"]["headlines_text"], splits["train"]["label_binary"])

    metrics = {}
    for name in ("train", "val", "test"):
        metrics[name] = tne.evaluate(model, splits[name], name)

    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": model.classifier,
                 "sentence_model_name": model.sentence_model_name,
                 "batch_size": model.batch_size,
                 "model_name": tne.MODEL_NAME,
                 "model_version": tne.MODEL_VERSION},
                MODELS_DIR / tne.MODEL_FILE)
    tne.save_predictions_csv(model, all_splits, PROCESSED_DATA_DIR / tne.CSV_NAME)
    tne.upsert_predictions_db(model, all_splits, db_path)
    return {"model": "news_embeddings", "metrics": metrics}


def _build_features_and_split() -> None:
    """Build LSTM feature sequences and create the chronological split."""
    from src.database.schema import DatabaseSchema
    from src.features.sequence_generator import SequenceGenerator
    from src.data_processing.dataset_split import DatasetSplitter

    DatabaseSchema().create_all_tables()

    gen = SequenceGenerator()
    all_X, all_y, all_dates = [], [], []
    for ticker in TICKERS:
        X, y, dates = gen.generate(ticker)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_dates.extend([(ticker, d) for d in dates])

    if not all_X:
        raise RuntimeError("No sequences generated — not enough price history.")

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / "X_sequences.npy", X_combined)
    np.save(PROCESSED_DATA_DIR / "y_labels.npy", y_combined)
    with open(PROCESSED_DATA_DIR / "sequence_dates.json", "w") as f:
        json.dump(all_dates, f)

    DatasetSplitter().split(train_ratio=0.70, val_ratio=0.15)
    log.info("Features built: %d sequences, split done.", len(X_combined))


def _run_lstm_training() -> dict:
    """Train the LSTM and export predictions to CSV + DB."""
    import train_lstm as tl

    X, y, dates_meta = tl.load_data()
    splits = tl.split_by_dates(X, y, dates_meta)

    X_train, y_train, _ = splits["train"]
    if len(X_train) == 0:
        raise RuntimeError("No LSTM training data.")

    from src.config import LSTM_CONFIG
    from src.models.lstm_model import StockLSTM, LSTMTrainer

    input_size = X_train.shape[2]
    model = StockLSTM(
        input_size=input_size,
        hidden_sizes=LSTM_CONFIG["lstm_units"],
        dropout=LSTM_CONFIG["dropout"],
    )
    trainer = LSTMTrainer(model, LSTM_CONFIG)
    X_val, y_val, _ = splits["val"]
    trainer.train(X_train, y_train, X_val, y_val)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save(MODELS_DIR / "lstm_model.pt")

    csv_path = PROCESSED_DATA_DIR / "price_predictions.csv"
    tl.save_predictions_csv(trainer, splits, csv_path)
    tl.upsert_predictions_db(trainer, splits, DATABASE_PATH)
    log.info("LSTM trained and predictions exported.")
    return {"model": "lstm_price"}


def _run_ensemble() -> None:
    """Build eval dataset then compute ensemble predictions.

    Calls the underlying functions directly instead of main() to avoid
    argparse conflicts with Flask's sys.argv.
    """
    import build_eval_dataset as bed
    import build_ensemble as be

    price_path = PROCESSED_DATA_DIR / "price_predictions.csv"
    tfidf_path = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"
    emb_path = PROCESSED_DATA_DIR / "news_embeddings_predictions.csv"
    eval_path = PROCESSED_DATA_DIR / "eval_dataset.csv"
    ensemble_path = PROCESSED_DATA_DIR / "final_ensemble_predictions.csv"

    # Step 1: build eval dataset (join the three prediction CSVs)
    price = bed.load_price(price_path)
    tfidf = bed.load_news_tfidf(tfidf_path)
    embeddings = bed.load_news_embeddings(emb_path)

    df = price.merge(tfidf, on=["ticker", "prediction_date"], how="inner")
    df = df.merge(embeddings, on=["ticker", "prediction_date"], how="inner")
    df["top_headlines"] = df.get("news_embeddings_top_headlines", pd.Series(dtype=str)).fillna(
        df.get("news_tfidf_top_headlines", "[]"))
    df["split"] = df.get("price_split", "unknown")

    out_cols = [c for c in bed.OUTPUT_COLS if c in df.columns]
    df = df[out_cols]
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(eval_path, index=False)

    # Step 2: compute ensemble
    eval_df = be.load_eval_dataset(eval_path)
    result = be.compute_ensemble(eval_df)
    result.to_csv(ensemble_path, index=False)

    log.info("Ensemble predictions built: %d rows.", len(result))


def _refresh_predictions() -> None:
    """Full pipeline: data → NLP → features → LSTM → ensemble."""
    global _refresh_status, _ensemble_cache
    results = []

    try:
        data_summary = _collect_fresh_data()
        log.info("Data collected: %s", data_summary)
    except Exception as exc:
        log.warning("Data collection failed: %s", exc)
        results.append({"model": "data_collection", "error": str(exc)})

    # NLP models
    _refresh_status["step"] = "Training TF-IDF model…"
    try:
        results.append(_run_tfidf_training())
    except Exception as exc:
        log.warning("TF-IDF training failed: %s", exc)
        results.append({"model": "news_tfidf", "error": str(exc)})

    _refresh_status["step"] = "Training embeddings model…"
    try:
        results.append(_run_embeddings_training())
    except Exception as exc:
        log.warning("Embeddings training skipped: %s", exc)
        results.append({"model": "news_embeddings", "error": str(exc)})

    # LSTM pipeline
    _refresh_status["step"] = "Building features…"
    try:
        _build_features_and_split()
    except Exception as exc:
        log.warning("Feature build failed: %s", exc)
        results.append({"model": "features", "error": str(exc)})

    _refresh_status["step"] = "Training LSTM model…"
    try:
        results.append(_run_lstm_training())
    except Exception as exc:
        log.warning("LSTM training failed: %s", exc)
        results.append({"model": "lstm_price", "error": str(exc)})

    # Ensemble
    _refresh_status["step"] = "Building ensemble…"
    try:
        _run_ensemble()
        _ensemble_cache = _load_ensemble_cache()
    except Exception as exc:
        log.warning("Ensemble build failed: %s", exc)
        results.append({"model": "ensemble", "error": str(exc)})

    with _refresh_lock:
        _refresh_status["running"] = False
        _refresh_status["step"] = None
        _refresh_status["last_date"] = str(date.today())
        errors = [r["error"] for r in results if "error" in r]
        _refresh_status["error"] = "; ".join(errors) if errors else None


# ═════════════════════════════════════════════════════════════════════════════
#  DATA ACCESS HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _native(val):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _latest_ensemble_prediction(ticker: str) -> dict | None:
    """Best available prediction for a ticker from the ensemble CSV.

    When NLP models have no signal (all 0.5), falls back to LSTM-only values.
    """
    edf = _ensemble_cache
    if edf is None or edf.empty:
        return None

    rows = edf[edf["ticker"] == ticker]
    if rows.empty:
        return None

    rows = rows.sort_values("prediction_date", ascending=False)
    row = rows.iloc[0]

    lstm_proba = row.get("financial_pred_proba")
    tfidf_proba = row.get("news_tfidf_pred_proba")
    emb_proba = row.get("news_embeddings_pred_proba")

    # Detect whether NLP has real signal
    nlp_has_signal = (
        (tfidf_proba is not None and round(float(tfidf_proba), 4) != 0.5)
        or (emb_proba is not None and round(float(emb_proba), 4) != 0.5)
    )

    if nlp_has_signal:
        # Use the ensemble formula as-is
        pred_proba = float(row["ensemble_pred_proba"])
        pred_binary = int(row["ensemble_pred_binary"])
        confidence = float(row["ensemble_confidence"])
        model_type = "ensemble"
    else:
        # NLP has no signal — use LSTM alone so we don't dilute it
        pred_proba = float(lstm_proba) if lstm_proba is not None and not pd.isna(lstm_proba) else 0.5
        pred_binary = int(pred_proba >= 0.5)
        confidence = abs(pred_proba - 0.5) * 2
        model_type = "lstm_only"

    actual = row.get("actual_binary")
    if actual is not None and not pd.isna(actual):
        actual = _native(int(actual))
    else:
        actual = None

    return {
        "ticker": _native(row["ticker"]),
        "date": _native(row["prediction_date"]),
        "pred_proba": _native(round(pred_proba, 4)),
        "pred_binary": _native(pred_binary),
        "confidence": _native(round(confidence, 4)),
        "model_type": model_type,
        "actual_binary": actual,
        "lstm_proba": _native(round(float(lstm_proba), 4)) if lstm_proba is not None and not pd.isna(lstm_proba) else None,
        "tfidf_proba": _native(round(float(tfidf_proba), 4)) if tfidf_proba is not None and not pd.isna(tfidf_proba) else None,
        "emb_proba": _native(round(float(emb_proba), 4)) if emb_proba is not None and not pd.isna(emb_proba) else None,
    }


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
    prediction = _latest_ensemble_prediction(ticker)
    news = _recent_news(conn, ticker)
    prices = _recent_prices(conn, ticker)
    label = _latest_label(conn, ticker)

    return {
        "ticker": ticker,
        "company": TICKER_TO_COMPANY.get(ticker, ticker),
        "prediction": prediction,
        "recent_news": news,
        "recent_prices": prices,
        "latest_label": label,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

def _scorecard(conn: sqlite3.Connection) -> dict:
    """Compare latest prediction direction vs actual outcome for each ticker."""
    correct, wrong, pending = 0, 0, 0
    details = []
    for ticker in TICKERS:
        pred = _latest_ensemble_prediction(ticker)
        label = _latest_label(conn, ticker)
        if pred is None or label is None:
            pending += 1
            continue
        pred_dir = pred["pred_binary"]
        actual_dir = label["label_binary"]
        hit = (pred_dir == actual_dir)
        if hit:
            correct += 1
        else:
            wrong += 1
        details.append({"ticker": ticker, "hit": hit})
    total = correct + wrong
    return {
        "correct": correct,
        "wrong": wrong,
        "pending": pending,
        "total": total,
        "pct": round(100 * correct / total, 1) if total else 0,
        "details": details,
    }


@app.route("/")
def index():
    q = request.args.get("q", "").strip().upper()
    conn = _get_db()
    try:
        selected = None
        if q and q in TICKERS:
            selected = _ticker_summary(conn, q)
        scorecard = _scorecard(conn)
        return render_template(
            "index.html",
            tickers=TICKERS,
            ticker_to_company=TICKER_TO_COMPANY,
            query=q,
            selected=selected,
            scorecard=scorecard,
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
