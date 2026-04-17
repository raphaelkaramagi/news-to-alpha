#!/usr/bin/env python3
"""News-to-Alpha interactive Flask app.

Serves the ensemble + per-model predictions from `final_ensemble_predictions.csv`
and exposes background-job endpoints for retraining / resetting models.

Routes
------
  GET  /                             -> configure landing page
  GET  /dashboard                    -> interactive dashboard
  GET  /admin                        -> training / reset controls
  GET  /healthz                      -> liveness
  GET  /api/ticker?ticker=AAPL[&model=ensemble|lstm|tfidf|embeddings]
  GET  /api/history?ticker=AAPL&window=30|90|180|all
  GET  /api/metrics
  GET  /api/presets                  -> available pipeline presets
  GET  /api/headlines?ticker=X&date=Y -> full headline cards with urls + sentiment
  GET  /api/rationale?ticker=X&date=Y -> meta-feature contributions for a call
  GET  /api/dates?ticker=X           -> available prediction dates for slider
  GET  /api/accuracy-trace?ticker=X&window=30 -> rolling accuracy for sparkline
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
from flask import Flask, jsonify, render_template, request

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

app = Flask(__name__, template_folder=str(_APP_DIR / "templates"))
jobs = JobRegistry(project_root=_PROJECT_ROOT)


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


def _latest_for_ticker(df: pd.DataFrame, ticker: str, model: str) -> Optional[LatestPrediction]:
    sub = df[df["ticker"] == ticker].copy()
    if sub.empty:
        return None

    sub = sub.sort_values("prediction_date")
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


@app.route("/api/ticker")
def api_ticker():
    ticker = (request.args.get("ticker") or "").upper()
    model = (request.args.get("model") or "ensemble").lower()
    if ticker not in TICKERS:
        return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
    if model not in ALLOWED_MODELS:
        return jsonify({"error": f"Unknown model: {model}"}), 400

    df = _load_predictions()
    if df is None:
        return jsonify({"error": "predictions not yet built"}), 503

    pred = _latest_for_ticker(df, ticker, model)
    if pred is None:
        return jsonify({"error": f"no predictions for {ticker}"}), 404

    return jsonify({
        "ticker": pred.ticker,
        "company": TICKER_TO_COMPANY.get(pred.ticker, pred.ticker),
        "prediction_date": pred.prediction_date,
        "model": pred.model,
        "proba": pred.proba,
        "binary": pred.binary,
        "confidence": pred.confidence,
        "actual_binary": pred.actual_binary,
        "top_headlines": pred.top_headlines,
        "per_model": pred.per_model,
    })


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
        if cfg_dict.get("skip_collect"):
            cmd_args += ["--skip-collect"]
        if cfg_dict.get("skip_news"):
            cmd_args += ["--skip-news"]
        rc = jobs.python_script(job, "run_pipeline.py", *cmd_args)
        if rc != 0:
            raise RuntimeError("run_pipeline.py failed")
    return run


@app.route("/api/presets")
def api_presets():
    from scripts.run_pipeline import PRESETS
    return jsonify({"presets": PRESETS})


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


@app.route("/api/rationale")
def api_rationale():
    """Return per-model probabilities + meta feature contributions for one call."""
    ticker = (request.args.get("ticker") or "").upper()
    date = (request.args.get("date") or "").strip()
    df = _load_predictions()
    if df is None:
        return jsonify({"error": "predictions not yet built"}), 503

    row_df = df[(df["ticker"] == ticker) & (df["prediction_date"] == date)]
    if row_df.empty:
        # fall back to latest date
        row_df = df[df["ticker"] == ticker].sort_values("prediction_date").tail(1)
    if row_df.empty:
        return jsonify({"error": f"no prediction for {ticker} on {date}"}), 404
    row = row_df.iloc[0]

    import joblib
    contributions: list[dict] = []
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

    for name in features:
        if name in row_df.columns and pd.notna(row[name]):
            contributions.append({
                "feature": name,
                "value": float(row[name]),
            })

    imp_map = {k: v for k, v in importances}
    for c in contributions:
        c["importance"] = float(imp_map.get(c["feature"], 0.0))
    contributions.sort(key=lambda x: abs(x["importance"]), reverse=True)

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
