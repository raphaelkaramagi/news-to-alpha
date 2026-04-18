#!/usr/bin/env python3
"""News-to-Alpha interactive Flask app.

Serves the ensemble + per-model predictions from `final_ensemble_predictions.csv`
and exposes background-job endpoints for retraining / resetting models.

Routes
------
  GET  /                             -> configure landing page
  GET  /dashboard                    -> interactive dashboard
  GET  /admin                        -> training / reset controls
  GET  /favicon.ico                  -> PNG favicon (tab icon; same bytes as 32×32 static file)
  GET  /static/icons/*               -> favicons, apple-touch-icon, web manifest
  GET  /healthz                      -> liveness
  GET  /api/data-status              -> freshness summary for header chip
  POST /api/data/refresh             -> fast refresh (collect + rebuild ensemble, no retraining)
  POST /api/data/clear               -> clear predictions | models | full reset
  GET  /api/ticker?ticker=AAPL[&model=ensemble|lstm|tfidf|embeddings][&date=YYYY-MM-DD]
  GET  /api/history?ticker=AAPL&window=30|90|180|all
  GET  /api/last-resolved?ticker=AAPL&n=7 -> last N rows with known actual_binary
  GET  /api/metrics
  GET  /api/presets                  -> available pipeline presets
  GET  /api/headlines?ticker=X&date=Y -> full headline cards with urls + sentiment
  GET  /api/rationale?ticker=X&date=Y -> meta-feature contributions for a call
  GET  /api/dates?ticker=X           -> available prediction dates for slider
  GET  /api/accuracy-trace?ticker=X&window=30 -> rolling accuracy for sparkline
  GET  /api/accuracy-summary?ticker=AAPL|ALL&window=7|30|90|all -> configurable recent accuracy
  GET  /api/conviction               -> accuracy by confidence bucket
  POST /api/run                      -> kick off full pipeline run
  POST /api/train                    -> kick off a per-model training job
  POST /api/reset                    -> remove a model's artifacts
  GET  /api/jobs                     -> current + recent jobs
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATABASE_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TICKER_TO_COMPANY,
    TICKERS,
)

from app.jobs import JobRegistry, JobSpec  # noqa: E402

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------

PREDICTIONS_CSV = PROCESSED_DATA_DIR / "final_ensemble_predictions.csv"
EVAL_OVERALL_CSV = PROCESSED_DATA_DIR / "evaluation_overall.csv"
EVAL_BY_TICKER_CSV = PROCESSED_DATA_DIR / "evaluation_by_ticker.csv"
EVAL_CONVICTION_CSV = PROCESSED_DATA_DIR / "evaluation_by_confidence.csv"
META_MODEL_PATH = MODELS_DIR / "ensemble_meta.joblib"

ALLOWED_MODELS = {"ensemble", "lstm", "tfidf", "embeddings"}
MODEL_PROBA_COL = {
    "ensemble":   "ensemble_pred_proba",
    "lstm":       "financial_pred_proba",
    "tfidf":      "news_tfidf_pred_proba",
    "embeddings": "news_embeddings_pred_proba",
}

# What `reset` wipes for each model.
MODEL_ARTIFACTS: dict[str, list[Path]] = {
    "lstm": [
        MODELS_DIR / "lstm_model.pt",
    ],
    "tfidf": [
        MODELS_DIR / "tfidf_lr.joblib",
        MODELS_DIR / "news_tfidf.joblib",
    ],
    "embeddings": [
        MODELS_DIR / "news_embeddings.joblib",
    ],
    "ensemble": [
        MODELS_DIR / "ensemble_meta.joblib",
        PREDICTIONS_CSV,
    ],
}


# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------

# Explicit paths so gunicorn/Docker always find templates + static (icons) regardless of cwd.
app = Flask(
    __name__,
    template_folder=str(_APP_DIR / "templates"),
    static_folder=str(_APP_DIR / "static"),
    static_url_path="/static",
)
jobs = JobRegistry(project_root=_PROJECT_ROOT)

# Bump when icons or manifest change so browsers/CDNs refetch (favicons are cached aggressively).
ASSET_VERSION = "2"


@app.context_processor
def _inject_asset_version() -> dict[str, str]:
    return {"asset_version": ASSET_VERSION}


@app.route("/favicon.ico")
def favicon_ico():
    """Serve PNG at the conventional path (no redirect — some clients mishandle 302 here)."""
    return send_from_directory(
        str(Path(app.static_folder) / "icons"),
        "favicon-32x32.png",
        mimetype="image/png",
        max_age=86400,
    )


# ----------------------------------------------------------------------------
# Data accessors
# ----------------------------------------------------------------------------

def _load_predictions() -> Optional[pd.DataFrame]:
    if not PREDICTIONS_CSV.exists():
        return None
    df = pd.read_csv(PREDICTIONS_CSV)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.strftime("%Y-%m-%d")
    return df


def _load_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {"overall": [], "by_ticker": []}
    if EVAL_OVERALL_CSV.exists():
        out["overall"] = pd.read_csv(EVAL_OVERALL_CSV).to_dict(orient="records")
    if EVAL_BY_TICKER_CSV.exists():
        out["by_ticker"] = pd.read_csv(EVAL_BY_TICKER_CSV).to_dict(orient="records")
    return out


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _recent_prices(ticker: str, days: Optional[int]) -> list[dict]:
    conn = _get_db()
    try:
        if days is None:
            rows = conn.execute(
                """SELECT date, open, high, low, close, volume
                   FROM prices WHERE ticker = ? ORDER BY date ASC""",
                (ticker,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT date, open, high, low, close, volume
                   FROM prices
                   WHERE ticker = ?
                   ORDER BY date DESC LIMIT ?""",
                (ticker, days),
            ).fetchall()
            rows = list(reversed(rows))
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _parse_window(value: str) -> Optional[int]:
    v = (value or "").strip().lower()
    if v in ("all", "max", ""):
        return None
    try:
        return max(1, int(v))
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# View helpers
# ----------------------------------------------------------------------------

@dataclass
class LatestPrediction:
    ticker: str
    prediction_date: str
    model: str
    proba: float
    binary: int
    confidence: float
    actual_binary: Optional[int]
    top_headlines: list[str]
    per_model: dict[str, dict[str, float]]


def _parse_top_headlines(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    s = str(raw).strip()
    if not s:
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [h.strip() for h in s.split("|") if h.strip()]


def _latest_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    model: str,
    date: Optional[str] = None,
) -> Optional[LatestPrediction]:
    sub = df[df["ticker"] == ticker].copy()
    if sub.empty:
        return None

    sub = sub.sort_values("prediction_date")
    if date:
        match = sub[sub["prediction_date"] == date]
        if match.empty:
            return None
        last = match.iloc[-1]
    else:
        last = sub.iloc[-1]

    col = MODEL_PROBA_COL[model]
    if col not in sub.columns:
        return None

    proba = float(last[col])
    binary = int(proba >= 0.5)
    confidence = abs(proba - 0.5) * 2.0

    per_model: dict[str, dict[str, float]] = {}
    for key, c in MODEL_PROBA_COL.items():
        if c in sub.columns and pd.notna(last[c]):
            p = float(last[c])
            per_model[key] = {
                "proba": p,
                "binary": int(p >= 0.5),
                "confidence": abs(p - 0.5) * 2.0,
            }

    actual = last.get("actual_binary")
    actual_int = int(actual) if pd.notna(actual) else None

    return LatestPrediction(
        ticker=ticker,
        prediction_date=str(last["prediction_date"]),
        model=model,
        proba=proba,
        binary=binary,
        confidence=confidence,
        actual_binary=actual_int,
        top_headlines=_parse_top_headlines(last.get("top_headlines")),
        per_model=per_model,
    )


# ----------------------------------------------------------------------------
# Jobs: runners
# ----------------------------------------------------------------------------

def _default_train_args(model: str, params: dict) -> list[str]:
    """Safely pass through allowed hyperparameters for each model."""
    args: list[str] = []

    def _num(name: str, cli: str, cast):
        if name in params and params[name] not in (None, ""):
            try:
                args.extend([cli, str(cast(params[name]))])
            except (TypeError, ValueError):
                pass

    if model == "lstm":
        _num("epochs", "--epochs", int)
        _num("batch_size", "--batch-size", int)
        _num("learning_rate", "--lr", float)
        _num("dropout", "--dropout", float)
        _num("seed", "--seed", int)
    elif model == "tfidf":
        _num("max_features", "--max-features", int)
        _num("C", "--C", float)
        _num("top_publishers", "--top-publishers", int)
    elif model == "embeddings":
        _num("C", "--C", float)
        _num("top_publishers", "--top-publishers", int)
        if params.get("use_finbert"):
            args.append("--use-finbert")
    return args


def _runner_train_lstm(params: dict):
    def run(job: JobSpec) -> None:
        if jobs.python_script(job, "train_lstm.py", *_default_train_args("lstm", params)) != 0:
            raise RuntimeError("train_lstm.py failed")
        if jobs.python_script(job, "build_eval_dataset.py") != 0:
            raise RuntimeError("build_eval_dataset.py failed")
        if jobs.python_script(job, "build_ensemble.py") != 0:
            raise RuntimeError("build_ensemble.py failed")
        jobs.python_script(job, "evaluate_predictions.py")
    return run


def _runner_train_tfidf(params: dict):
    def run(job: JobSpec) -> None:
        if jobs.python_script(job, "train_nlp.py", *_default_train_args("tfidf", params)) != 0:
            raise RuntimeError("train_nlp.py failed")
        if jobs.python_script(job, "build_eval_dataset.py") != 0:
            raise RuntimeError("build_eval_dataset.py failed")
        if jobs.python_script(job, "build_ensemble.py") != 0:
            raise RuntimeError("build_ensemble.py failed")
        jobs.python_script(job, "evaluate_predictions.py")
    return run


def _runner_train_embeddings(params: dict):
    def run(job: JobSpec) -> None:
        if jobs.python_script(
            job, "train_news_embeddings.py", *_default_train_args("embeddings", params)
        ) != 0:
            raise RuntimeError("train_news_embeddings.py failed")
        if jobs.python_script(job, "build_eval_dataset.py") != 0:
            raise RuntimeError("build_eval_dataset.py failed")
        if jobs.python_script(job, "build_ensemble.py") != 0:
            raise RuntimeError("build_ensemble.py failed")
        jobs.python_script(job, "evaluate_predictions.py")
    return run


def _runner_train_ensemble(_params: dict):
    def run(job: JobSpec) -> None:
        if jobs.python_script(job, "build_eval_dataset.py") != 0:
            raise RuntimeError("build_eval_dataset.py failed")
        if jobs.python_script(job, "build_ensemble.py") != 0:
            raise RuntimeError("build_ensemble.py failed")
        jobs.python_script(job, "evaluate_predictions.py")
    return run


def _runner_train_all(params: dict):
    def run(job: JobSpec) -> None:
        _runner_train_lstm(params)(job)  # already runs build_eval + ensemble
        _runner_train_tfidf(params)(job)
        _runner_train_embeddings(params)(job)
    return run


def _runner_reset(model: str):
    def run(job: JobSpec) -> None:
        removed = 0
        for p in MODEL_ARTIFACTS.get(model, []):
            if p.exists():
                p.unlink()
                job.log.append(f"removed {p.relative_to(_PROJECT_ROOT)}")
                removed += 1
        job.log.append(f"reset complete; {removed} artifact(s) removed for {model}")
    return run


TRAIN_RUNNERS = {
    "lstm": _runner_train_lstm,
    "tfidf": _runner_train_tfidf,
    "embeddings": _runner_train_embeddings,
    "ensemble": _runner_train_ensemble,
    "all": _runner_train_all,
}


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def configure_page():
    """New landing / configure page."""
    from scripts.run_pipeline import PRESETS
    predictions_exist = PREDICTIONS_CSV.exists()
    return render_template(
        "configure.html",
        tickers=TICKERS,
        ticker_to_company=TICKER_TO_COMPANY,
        presets=PRESETS,
        predictions_exist=predictions_exist,
    )


@app.route("/dashboard")
def dashboard():
    df = _load_predictions()
    if df is None:
        return render_template(
            "dashboard.html",
            tickers=TICKERS,
            ticker_to_company=TICKER_TO_COMPANY,
            missing_predictions=True,
        )
    latest_date = df["prediction_date"].max()
    return render_template(
        "dashboard.html",
        tickers=TICKERS,
        ticker_to_company=TICKER_TO_COMPANY,
        missing_predictions=False,
        latest_date=latest_date,
        default_ticker=TICKERS[0] if TICKERS else "",
    )


@app.route("/admin")
def admin():
    return render_template(
        "admin.html",
        tickers=TICKERS,
        ticker_to_company=TICKER_TO_COMPANY,
    )


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "predictions_csv": PREDICTIONS_CSV.exists(),
        "metrics_csv": EVAL_OVERALL_CSV.exists(),
        "tickers": len(TICKERS),
    })


@app.route("/api/data-status")
def api_data_status():
    """Freshness summary - powers the header 'Data to:' chip + the reconfigure drawer."""
    df = _load_predictions()
    latest_pred = None
    rows = 0
    if df is not None and not df.empty:
        latest_pred = str(df["prediction_date"].max())
        rows = int(len(df))

    latest_price = None
    latest_news = None
    price_count = 0
    news_count = 0
    try:
        conn = _get_db()
        try:
            row = conn.execute("SELECT MAX(date) AS d, COUNT(*) AS n FROM prices").fetchone()
            if row:
                latest_price = str(row["d"]) if row["d"] else None
                price_count = int(row["n"] or 0)
            row = conn.execute(
                "SELECT MAX(date(published_at)) AS d, COUNT(*) AS n FROM news"
            ).fetchone()
            if row:
                latest_news = str(row["d"]) if row["d"] else None
                news_count = int(row["n"] or 0)
        finally:
            conn.close()
    except Exception:
        pass

    from datetime import date as _date
    today = _date.today().isoformat()
    return jsonify({
        "today": today,
        "latest_prediction_date": latest_pred,
        "latest_price_date": latest_price,
        "latest_news_date": latest_news,
        "prediction_rows": rows,
        "price_rows": price_count,
        "news_rows": news_count,
    })


@app.route("/api/ticker")
def api_ticker():
    ticker = (request.args.get("ticker") or "").upper()
    model = (request.args.get("model") or "ensemble").lower()
    date = (request.args.get("date") or "").strip() or None
    if ticker not in TICKERS:
        return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
    if model not in ALLOWED_MODELS:
        return jsonify({"error": f"Unknown model: {model}"}), 400

    df = _load_predictions()
    if df is None:
        return jsonify({"error": "predictions not yet built"}), 503

    pred = _latest_for_ticker(df, ticker, model, date=date)
    if pred is None:
        if date:
            return jsonify({"error": f"no prediction for {ticker} on {date}"}), 404
        return jsonify({"error": f"no predictions for {ticker}"}), 404

    realized_return = None
    hit: Optional[int] = None
    if pred.actual_binary is not None:
        hit = int(pred.binary == pred.actual_binary)
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT label_return FROM labels WHERE ticker = ? AND date = ?",
                (pred.ticker, pred.prediction_date),
            ).fetchone()
            if row and row["label_return"] is not None:
                realized_return = float(row["label_return"])
        finally:
            conn.close()

    return jsonify({
        "ticker": pred.ticker,
        "company": TICKER_TO_COMPANY.get(pred.ticker, pred.ticker),
        "prediction_date": pred.prediction_date,
        "model": pred.model,
        "proba": pred.proba,
        "binary": pred.binary,
        "confidence": pred.confidence,
        "actual_binary": pred.actual_binary,
        "hit": hit,
        "realized_return": realized_return,
        "top_headlines": pred.top_headlines,
        "per_model": pred.per_model,
    })


@app.route("/api/last-resolved")
def api_last_resolved():
    """Return the most recent N predictions for a ticker that have a known actual.

    Powers the 'Yesterday vs Actual' card and the 7-dot history strip.
    """
    ticker = (request.args.get("ticker") or "").upper()
    try:
        n = max(1, min(60, int(request.args.get("n") or "7")))
    except ValueError:
        n = 7
    if ticker not in TICKERS:
        return jsonify({"error": f"Unknown ticker: {ticker}"}), 404

    df = _load_predictions()
    if df is None:
        return jsonify({"ticker": ticker, "rows": []})

    sub = df[df["ticker"] == ticker].copy()
    sub = sub[sub["actual_binary"].notna()].sort_values("prediction_date")
    if sub.empty:
        return jsonify({"ticker": ticker, "rows": []})

    sub = sub.tail(n)

    # Pull realized returns from labels table (keyed on ticker, date == prediction_date)
    returns_by_date: dict[str, float] = {}
    conn = _get_db()
    try:
        placeholders = ",".join(["?"] * len(sub))
        rows = conn.execute(
            f"SELECT date, label_return FROM labels "
            f"WHERE ticker = ? AND date IN ({placeholders})",
            (ticker, *sub["prediction_date"].tolist()),
        ).fetchall()
        for r in rows:
            if r["label_return"] is not None:
                returns_by_date[str(r["date"])] = float(r["label_return"])
    finally:
        conn.close()

    out: list[dict] = []
    for _, row in sub.iterrows():
        proba = float(row["ensemble_pred_proba"]) if pd.notna(row.get("ensemble_pred_proba")) else None
        pred_bin = int(row["ensemble_pred_binary"]) if pd.notna(row.get("ensemble_pred_binary")) else None
        actual = int(row["actual_binary"]) if pd.notna(row.get("actual_binary")) else None
        hit = None
        if pred_bin is not None and actual is not None:
            hit = int(pred_bin == actual)
        date_str = str(row["prediction_date"])
        out.append({
            "date": date_str,
            "proba": proba,
            "pred_binary": pred_bin,
            "actual_binary": actual,
            "hit": hit,
            "return": returns_by_date.get(date_str),
        })
    return jsonify({"ticker": ticker, "rows": out})


@app.route("/api/history")
def api_history():
    ticker = (request.args.get("ticker") or "").upper()
    window = _parse_window(request.args.get("window") or "90")
    if ticker not in TICKERS:
        return jsonify({"error": f"Unknown ticker: {ticker}"}), 404

    df = _load_predictions()
    preds_records: list[dict] = []
    if df is not None:
        sub = df[df["ticker"] == ticker].sort_values("prediction_date")
        if window is not None:
            sub = sub.tail(window)
        cols = ["prediction_date", "ensemble_pred_proba", "financial_pred_proba",
                "news_tfidf_pred_proba", "news_embeddings_pred_proba",
                "ensemble_pred_binary", "actual_binary", "has_news"]
        existing = [c for c in cols if c in sub.columns]
        preds_records = sub[existing].where(pd.notna(sub[existing]), None).to_dict(orient="records")

    prices = _recent_prices(ticker, days=(window if window else None))

    return jsonify({
        "ticker": ticker,
        "window": window or "all",
        "prices": prices,
        "predictions": preds_records,
    })


@app.route("/api/metrics")
def api_metrics():
    return jsonify(_load_metrics())


@app.route("/api/jobs")
def api_jobs():
    return jsonify({
        "current": jobs.current(),
        "recent": jobs.recent(n=5),
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(silent=True) or {}
    model = str(payload.get("model") or "").lower()
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        return jsonify({"error": "params must be an object"}), 400
    if model not in TRAIN_RUNNERS:
        return jsonify({
            "error": f"model must be one of {sorted(TRAIN_RUNNERS)}",
        }), 400

    runner = TRAIN_RUNNERS[model](params)
    accepted, job = jobs.submit(
        kind=f"train_{model}",
        label=f"Retraining {model}",
        runner=runner,
    )
    status = 202 if accepted else 409
    return jsonify({"accepted": accepted, "job": job}), status


@app.route("/api/reset", methods=["POST"])
def api_reset():
    payload = request.get_json(silent=True) or {}
    model = str(payload.get("model") or "").lower()
    if model not in MODEL_ARTIFACTS:
        return jsonify({
            "error": f"model must be one of {sorted(MODEL_ARTIFACTS)}",
        }), 400

    accepted, job = jobs.submit(
        kind=f"reset_{model}",
        label=f"Resetting {model}",
        runner=_runner_reset(model),
    )
    status = 202 if accepted else 409
    return jsonify({"accepted": accepted, "job": job}), status


# ----------------------------------------------------------------------------
# New API routes (run pipeline, headlines, rationale, dates, etc.)
# ----------------------------------------------------------------------------


def _runner_run_pipeline(cfg_dict: dict):
    """Submit a full pipeline run via scripts/run_pipeline.py."""
    def run(job: JobSpec) -> None:
        cmd_args: list[str] = []
        tickers = cfg_dict.get("tickers") or []
        if tickers:
            cmd_args += ["--tickers", *tickers]
        if cfg_dict.get("lookback_days"):
            cmd_args += ["--lookback-days", str(int(cfg_dict["lookback_days"]))]
        if cfg_dict.get("horizon"):
            cmd_args += ["--horizon", str(int(cfg_dict["horizon"]))]
        if cfg_dict.get("min_move_pct") is not None:
            cmd_args += ["--min-move-pct", str(float(cfg_dict["min_move_pct"]))]
        seeds = cfg_dict.get("seeds") or []
        if seeds:
            cmd_args += ["--seeds", *[str(int(s)) for s in seeds]]
        if cfg_dict.get("use_finbert"):
            cmd_args += ["--use-finbert"]
        for flag_name, cli in [
            ("skip_collect",  "--skip-collect"),
            ("skip_news",     "--skip-news"),
            ("skip_labels",   "--skip-labels"),
            ("skip_split",    "--skip-split"),
            ("skip_lstm",     "--skip-lstm"),
            ("skip_nlp",      "--skip-nlp"),
            ("skip_emb",      "--skip-emb"),
            ("skip_ensemble", "--skip-ensemble"),
            ("skip_evaluate", "--skip-evaluate"),
        ]:
            if cfg_dict.get(flag_name):
                cmd_args += [cli]
        rc = jobs.python_script(job, "run_pipeline.py", *cmd_args)
        if rc != 0:
            raise RuntimeError("run_pipeline.py failed")
    return run


@app.route("/api/presets")
def api_presets():
    from scripts.run_pipeline import PRESETS
    return jsonify({"presets": PRESETS})


def _runner_data_refresh(days: int, include_news: bool = True,
                         mode: str = "quality"):
    """Collect latest prices + news and re-score the whole pipeline.

    Modes:
      - ``quality`` (default): 3-seed LSTM averaging (~3 min). Confidence
        numbers come out smooth and informative.
      - ``fast``:              1-seed LSTM (~1 min). Good enough to advance
        dates but confidence will be blocky.

    We delegate to ``run_pipeline.py`` because every base model
    (LSTM / TF-IDF / embeddings) only writes prediction rows for
    (ticker, date) pairs it was *run* over - so just rebuilding the
    ensemble off stale per-model CSVs would keep the dashboard
    frozen at whatever date training last touched.
    """
    seeds = ["42", "1337", "2024"] if str(mode).lower() != "fast" else ["42"]

    def run(job: JobSpec) -> None:
        cmd_args: list[str] = [
            "--tickers", *TICKERS,
            "--lookback-days", str(int(days)),
            "--seeds", *seeds,
        ]
        if not include_news:
            cmd_args.append("--skip-news")
        rc = jobs.python_script(job, "run_pipeline.py", *cmd_args)
        if rc != 0:
            raise RuntimeError("run_pipeline.py failed")
    return run


def _runner_data_clear(scope: str):
    """Clear various data artifacts.

    scope="predictions"  - just delete final_ensemble_predictions.csv (safe)
    scope="models"       - delete model artifacts (prompts re-train)
    scope="full"         - delete DB, models, processed (keeps raw downloads)
    """
    def run(job: JobSpec) -> None:
        import shutil
        removed = []
        if scope in ("predictions", "models", "full"):
            if PREDICTIONS_CSV.exists():
                PREDICTIONS_CSV.unlink()
                removed.append(str(PREDICTIONS_CSV.relative_to(_PROJECT_ROOT)))
        if scope in ("models", "full"):
            for paths in MODEL_ARTIFACTS.values():
                for p in paths:
                    if p.exists():
                        p.unlink()
                        removed.append(str(p.relative_to(_PROJECT_ROOT)))
            meta = MODELS_DIR / "ensemble_meta.joblib"
            if meta.exists():
                meta.unlink()
                removed.append(str(meta.relative_to(_PROJECT_ROOT)))
        if scope == "full":
            if DATABASE_PATH.exists():
                DATABASE_PATH.unlink()
                removed.append(str(DATABASE_PATH.relative_to(_PROJECT_ROOT)))
            if PROCESSED_DATA_DIR.exists():
                shutil.rmtree(PROCESSED_DATA_DIR)
                removed.append(str(PROCESSED_DATA_DIR.relative_to(_PROJECT_ROOT)) + "/")
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for r in removed:
            job.log.append(f"removed {r}")
        job.log.append(f"clear complete ({scope}); {len(removed)} path(s) removed")
    return run


@app.route("/api/data/refresh", methods=["POST"])
def api_data_refresh():
    """Kick off a 'fast' refresh: collect new prices+news, rebuild ensemble, no retraining."""
    payload = request.get_json(silent=True) or {}
    try:
        days = max(1, int(payload.get("days") or 30))
    except (TypeError, ValueError):
        days = 30
    include_news = bool(payload.get("include_news", True))
    mode = str(payload.get("mode") or "quality").lower()
    if mode not in ("quality", "fast"):
        mode = "quality"

    accepted, job = jobs.submit(
        kind="data_refresh",
        label=f"Refreshing data (last {days}d, {mode})",
        runner=_runner_data_refresh(
            days=days, include_news=include_news, mode=mode,
        ),
    )
    status = 202 if accepted else 409
    return jsonify({"accepted": accepted, "job": job}), status


@app.route("/api/data/clear", methods=["POST"])
def api_data_clear():
    """Clear artifacts. scope = predictions | models | full."""
    payload = request.get_json(silent=True) or {}
    scope = str(payload.get("scope") or "predictions").lower()
    if scope not in ("predictions", "models", "full"):
        return jsonify({"error": "scope must be predictions | models | full"}), 400

    accepted, job = jobs.submit(
        kind=f"data_clear_{scope}",
        label=f"Clearing {scope}",
        runner=_runner_data_clear(scope),
    )
    status = 202 if accepted else 409
    return jsonify({"accepted": accepted, "job": job}), status


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(silent=True) or {}
    preset = str(payload.get("preset") or "").lower()
    config = payload.get("config") or {}
    if not isinstance(config, dict):
        return jsonify({"error": "config must be an object"}), 400

    from scripts.run_pipeline import PRESETS
    base = PRESETS.get(preset, {}).copy()
    base.update({k: v for k, v in config.items() if v not in (None, "")})

    # sanitize
    if base.get("tickers") is None:
        base["tickers"] = TICKERS
    base["tickers"] = [str(t).upper() for t in (base.get("tickers") or [])]

    accepted, job = jobs.submit(
        kind="pipeline_run",
        label=f"Pipeline ({preset or 'custom'})",
        runner=_runner_run_pipeline(base),
    )
    status = 202 if accepted else 409
    return jsonify({"accepted": accepted, "job": job, "config": base}), status


def _load_headlines_for(ticker: str, date: str) -> list[dict]:
    from src.models.news_pipeline import map_published_to_label_date
    conn = _get_db()
    try:
        cols_info = {
            row["name"] for row in conn.execute("PRAGMA table_info(news)").fetchall()
        }
        col_content = "content" if "content" in cols_info else "NULL AS content"
        col_sent = (
            "sentiment_score" if "sentiment_score" in cols_info
            else "NULL AS sentiment_score"
        )
        col_rel = (
            "relevance_score" if "relevance_score" in cols_info
            else "NULL AS relevance_score"
        )
        col_url = "url" if "url" in cols_info else "NULL AS url"

        rows = conn.execute(
            f"SELECT ticker, title, source, published_at, {col_content}, "
            f"{col_sent}, {col_rel}, {col_url} "
            "FROM news WHERE ticker = ? AND title IS NOT NULL AND title != '' "
            "ORDER BY published_at DESC",
            (ticker,),
        ).fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        label_date = map_published_to_label_date(r["published_at"])
        if label_date != date:
            continue
        out.append({
            "title": r["title"],
            "source": r["source"] or "unknown",
            "published_at": r["published_at"],
            "content": (r["content"] or "")[:1200] if "content" in r.keys() else "",
            "sentiment_finnhub": (
                float(r["sentiment_score"]) if r["sentiment_score"] is not None else None
            ),
            "relevance": (
                float(r["relevance_score"]) if r["relevance_score"] is not None else None
            ),
            "url": r["url"],
        })
    return out


@app.route("/api/headlines")
def api_headlines():
    ticker = (request.args.get("ticker") or "").upper()
    date = (request.args.get("date") or "").strip()
    if not ticker or not date:
        return jsonify({"error": "ticker and date are required"}), 400
    try:
        items = _load_headlines_for(ticker, date)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"ticker": ticker, "date": date, "headlines": items})


@app.route("/api/dates")
def api_dates():
    ticker = (request.args.get("ticker") or "").upper()
    df = _load_predictions()
    if df is None:
        return jsonify({"ticker": ticker, "dates": []})
    sub = df if not ticker else df[df["ticker"] == ticker]
    dates = sorted(sub["prediction_date"].unique().tolist())
    return jsonify({"ticker": ticker, "dates": dates})


# Baselines for local contribution. Each feature has a neutral point; the contribution
# shown in Why-this-call is importance * (value - baseline), scaled to [-1, 1].
# This makes bars vary per-day (unlike raw global importance) and gives a sign
# indicating whether the feature pushed the ensemble UP (+) or DOWN (-).
_FEATURE_BASELINES: dict[str, float] = {
    "financial_pred_proba":       0.5,
    "news_tfidf_pred_proba":      0.5,
    "news_embeddings_pred_proba": 0.5,
    "lstm_confidence":            0.0,
    "tfidf_confidence":           0.0,
    "emb_confidence":             0.0,
    "all_agree":                  0.5,
    "has_news":                   0.5,
    "n_headlines":                0.0,  # overwritten with median below
    "spy_return_5d":              0.0,
}


def _compute_feature_scales(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    """Return per-feature {baseline, scale} for normalising contribution bars."""
    out: dict[str, dict[str, float]] = {}
    for f in features:
        if f not in df.columns:
            continue
        series = pd.to_numeric(df[f], errors="coerce").dropna()
        if series.empty:
            continue
        baseline = _FEATURE_BASELINES.get(f, float(series.median()))
        if f == "n_headlines":
            baseline = float(series.median())
        # scale = interquartile range with a small floor; avoids div-by-zero
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        scale = float(max(q3 - q1, series.std() or 0.0, 1e-6))
        out[f] = {"baseline": float(baseline), "scale": scale}
    return out


@app.route("/api/rationale")
def api_rationale():
    """Return per-model probabilities + local feature contributions for one call.

    Bars are driven by signed *local* contribution:
        contrib = importance * (value - baseline) / scale
    so the same feature shifts position and sign across days.
    """
    ticker = (request.args.get("ticker") or "").upper()
    date = (request.args.get("date") or "").strip()
    df = _load_predictions()
    if df is None:
        return jsonify({"error": "predictions not yet built"}), 503

    row_df = df[(df["ticker"] == ticker) & (df["prediction_date"] == date)]
    if row_df.empty:
        row_df = df[df["ticker"] == ticker].sort_values("prediction_date").tail(1)
    if row_df.empty:
        return jsonify({"error": f"no prediction for {ticker} on {date}"}), 404
    row = row_df.iloc[0]

    import joblib
    features: list[str] = []
    importances: list[tuple[str, float]] = []
    temperature = 1.0
    if META_MODEL_PATH.exists():
        try:
            payload = joblib.load(META_MODEL_PATH)
            features = payload.get("features", [])
            importances = payload.get("importances", []) or []
            temperature = float(payload.get("temperature", 1.0))
        except Exception:
            features, importances = [], []

    imp_map = {k: v for k, v in importances}
    scales = _compute_feature_scales(df, features)

    contributions: list[dict] = []
    for name in features:
        if name not in row_df.columns or pd.isna(row[name]):
            continue
        value = float(row[name])
        imp = float(imp_map.get(name, 0.0))
        sc = scales.get(name, {"baseline": 0.0, "scale": 1.0})
        baseline = sc["baseline"]
        scale = sc["scale"] if sc["scale"] > 0 else 1.0
        # Signed local contribution: how much this feature nudged the call,
        # relative to what the model considers a neutral value for it.
        contribution = imp * (value - baseline) / scale
        contributions.append({
            "feature": name,
            "value": value,
            "importance": imp,
            "baseline": baseline,
            "contribution": contribution,
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    per_model = {}
    for name, col in MODEL_PROBA_COL.items():
        if col in row_df.columns and pd.notna(row[col]):
            p = float(row[col])
            per_model[name] = {
                "proba": p,
                "confidence": abs(p - 0.5) * 2.0,
                "binary": int(p >= 0.5),
            }

    return jsonify({
        "ticker": ticker,
        "date": str(row["prediction_date"]),
        "per_model": per_model,
        "ensemble_proba": float(row.get("ensemble_pred_proba", 0.5)) if "ensemble_pred_proba" in row_df.columns else None,
        "ensemble_confidence": float(row.get("ensemble_confidence", 0.0)) if "ensemble_confidence" in row_df.columns else None,
        "contributions": contributions[:8],
        "temperature": temperature,
    })


@app.route("/api/accuracy-trace")
def api_accuracy_trace():
    ticker = (request.args.get("ticker") or "").upper()
    try:
        window = max(5, int(request.args.get("window") or "30"))
    except ValueError:
        window = 30

    df = _load_predictions()
    if df is None:
        return jsonify({"ticker": ticker, "window": window, "series": []})

    sub = df[df["ticker"] == ticker].copy() if ticker else df.copy()
    sub = sub[sub["actual_binary"].notna()].sort_values("prediction_date")
    if sub.empty:
        return jsonify({"ticker": ticker, "window": window, "series": []})

    sub["correct"] = (sub["ensemble_pred_binary"] == sub["actual_binary"]).astype(int)
    sub["rolling_acc"] = sub["correct"].rolling(window=window, min_periods=5).mean()

    series = [
        {"date": d, "accuracy": (float(a) if pd.notna(a) else None)}
        for d, a in zip(sub["prediction_date"], sub["rolling_acc"])
    ]
    return jsonify({"ticker": ticker, "window": window, "series": series})


@app.route("/api/accuracy-summary")
def api_accuracy_summary():
    """Aggregate ensemble accuracy over a recent window.

    Query:
      ticker=AAPL | ALL   (defaults to ALL)
      window=7 | 30 | 90 | all
    Returns:
      { scope, window, n, hits, accuracy,
        rows: [{date, ticker, pred_binary, actual_binary, hit, return}, ...] }
    """
    ticker = (request.args.get("ticker") or "ALL").upper()
    window_raw = (request.args.get("window") or "30").lower()
    df = _load_predictions()
    if df is None:
        return jsonify({"scope": ticker, "window": window_raw, "n": 0, "rows": []})

    sub = df if ticker == "ALL" else df[df["ticker"] == ticker]
    sub = sub[sub["actual_binary"].notna()].copy()
    if sub.empty:
        return jsonify({"scope": ticker, "window": window_raw, "n": 0, "rows": []})

    sub = sub.sort_values("prediction_date")
    # Parse window: integer = last-N resolved dates (per-ticker intuition), "all" = everything
    if window_raw == "all":
        window_trim = sub
    else:
        try:
            n = max(1, int(window_raw))
        except ValueError:
            n = 30
        # For ALL, take the last N *trading dates*, not last N rows.
        if ticker == "ALL":
            latest_dates = sorted(sub["prediction_date"].unique())[-n:]
            window_trim = sub[sub["prediction_date"].isin(latest_dates)]
        else:
            window_trim = sub.tail(n)

    window_trim = window_trim.copy()
    window_trim["hit"] = (
        window_trim["ensemble_pred_binary"] == window_trim["actual_binary"]
    ).astype(int)

    # Pull realized returns from labels table if available
    conn = _get_db()
    try:
        label_rows = conn.execute(
            "SELECT ticker, date, label_return FROM labels"
        ).fetchall()
    finally:
        conn.close()
    label_map: dict[tuple[str, str], float] = {}
    for lr in label_rows:
        if lr["label_return"] is not None:
            label_map[(lr["ticker"], str(lr["date"]))] = float(lr["label_return"])

    rows = []
    for _, r in window_trim.iterrows():
        key = (r["ticker"], str(r["prediction_date"]))
        rows.append({
            "date": str(r["prediction_date"]),
            "ticker": r["ticker"],
            "pred_binary": int(r["ensemble_pred_binary"]) if pd.notna(r["ensemble_pred_binary"]) else None,
            "actual_binary": int(r["actual_binary"]) if pd.notna(r["actual_binary"]) else None,
            "hit": int(r["hit"]),
            "proba": float(r["ensemble_pred_proba"]) if pd.notna(r.get("ensemble_pred_proba")) else None,
            "return": label_map.get(key),
        })

    n = len(rows)
    hits = int(sum(r["hit"] for r in rows))
    return jsonify({
        "scope": ticker,
        "window": window_raw,
        "n": n,
        "hits": hits,
        "accuracy": (hits / n) if n else None,
        "rows": rows,
    })


@app.route("/api/conviction")
def api_conviction():
    if not EVAL_CONVICTION_CSV.exists():
        return jsonify({"buckets": []})
    df = pd.read_csv(EVAL_CONVICTION_CSV)
    return jsonify({"buckets": df.to_dict(orient="records")})


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _create_app() -> Flask:
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News-to-Alpha Flask app")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n  News-to-Alpha running at http://{args.host}:{args.port}")
    print(f"  Predictions CSV : {PREDICTIONS_CSV}")
    print(f"  Database        : {DATABASE_PATH}\n")
    app.run(host=args.host, port=args.port, debug=args.debug)
