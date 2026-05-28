#!/usr/bin/env python3
"""News-to-Alpha JSON API backend.

Serves predictions from `final_ensemble_predictions.csv` and SQLite.
The public UI is the Next.js app in `web/` (Vercel); this process is API-only.

Key routes: `/healthz`, `/api/data-status`, `/api/ticker`, `/api/history`,
`/api/headlines`, `/api/rationale`, `/api/dates`, etc.

Training / pipeline mutations (`POST /api/run`, `/api/train`, …) are disabled
when `INFERENCE_ONLY=true` (Railway production).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from flask import Flask, Response, jsonify, request

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
from src.utils.pipeline_config import load_or_default as _load_pipeline_cfg  # noqa: E402
from src.utils.trading_calendar import (  # noqa: E402
    last_trading_session,
    prediction_lag_sessions,
)

from app.jobs import JobRegistry, JobSpec  # noqa: E402

# Runtime mode (Railway lean deployment)
_INFERENCE_ONLY = os.getenv("INFERENCE_ONLY", "false").lower() in ("true", "1", "yes")

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

_LSTM_THRESHOLD_CACHE: float | None = None


def _lstm_decision_threshold() -> float:
    global _LSTM_THRESHOLD_CACHE
    if _LSTM_THRESHOLD_CACHE is not None:
        return _LSTM_THRESHOLD_CACHE
    path = MODELS_DIR / "lstm_model.pt"
    if not path.exists():
        _LSTM_THRESHOLD_CACHE = 0.5
        return 0.5
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        _LSTM_THRESHOLD_CACHE = float(ckpt.get("decision_threshold", 0.5))
    except Exception:
        _LSTM_THRESHOLD_CACHE = 0.5
    return _LSTM_THRESHOLD_CACHE


def _model_pred_binary(df: pd.DataFrame, model: str) -> pd.Series:
    """Binary predictions for a model head (uses LSTM tuned threshold when applicable)."""
    model = model.lower()
    proba_col = MODEL_PROBA_COL.get(model, "ensemble_pred_proba")
    if proba_col not in df.columns:
        return pd.Series(0, index=df.index)
    proba = df[proba_col].fillna(0.5).astype(float)
    if model == "ensemble" and "ensemble_pred_binary" in df.columns:
        return df["ensemble_pred_binary"].fillna((proba >= 0.5).astype(int)).astype(int)
    threshold = _lstm_decision_threshold() if model == "lstm" else 0.5
    return (proba >= threshold).astype(int)

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

app = Flask(__name__)
jobs = JobRegistry(project_root=_PROJECT_ROOT)


# ----------------------------------------------------------------------------
# Data accessors
# ----------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/Inf so JSON is valid in browsers (JSON.parse rejects NaN)."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _jsonify_safe(payload: Any, status: int = 200) -> Response:
    return Response(
        json.dumps(_json_safe(payload)),
        status=status,
        mimetype="application/json",
    )


def _df_records(df: pd.DataFrame, cols: Optional[list[str]] = None) -> list[dict]:
    """DataFrame → JSON-safe records (NaN → null)."""
    sub = df[cols] if cols else df
    cleaned = sub.where(pd.notna(sub), None)
    return _json_safe(cleaned.to_dict(orient="records"))


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


def _price_context(ticker: str, session_date: str) -> dict[str, Any]:
    """OHLC context for a forecast keyed on session T (close T → close T+1)."""
    from datetime import date as _date

    from src.utils.trading_calendar import next_trading_session

    try:
        session = _date.fromisoformat(session_date)
    except ValueError:
        return {"session_date": session_date, "target_date": None}

    target_str = next_trading_session(session).isoformat()
    conn = _get_db()
    try:
        price_rows = conn.execute(
            """SELECT date, open, close FROM prices
               WHERE ticker = ? AND date IN (?, ?)""",
            (ticker, session_date, target_str),
        ).fetchall()
        label_row = conn.execute(
            """SELECT close_t, close_t_plus_1, label_return, label_binary
               FROM labels WHERE ticker = ? AND date = ?""",
            (ticker, session_date),
        ).fetchone()
    finally:
        conn.close()

    by_date = {str(r["date"]): dict(r) for r in price_rows}
    sess = by_date.get(session_date, {})
    targ = by_date.get(target_str, {})

    session_close = sess.get("close")
    if session_close is None and label_row and label_row["close_t"] is not None:
        session_close = float(label_row["close_t"])
    target_close = targ.get("close")
    if target_close is None and label_row and label_row["close_t_plus_1"] is not None:
        target_close = float(label_row["close_t_plus_1"])

    label_return = None
    if label_row and label_row["label_return"] is not None:
        label_return = float(label_row["label_return"])
    elif session_close is not None and target_close is not None and session_close:
        label_return = (target_close / session_close - 1.0) * 100.0

    resolved = target_close is not None
    actual_direction: Optional[str] = None
    if label_row and label_row["label_binary"] is not None:
        actual_direction = "up" if int(label_row["label_binary"]) == 1 else "down"

    def _num(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    return {
        "session_date": session_date,
        "target_date": target_str,
        # Primary fields used for scoring (close T → close T+1)
        "start_close_date": session_date,
        "start_close": _num(session_close),
        "end_close_date": target_str,
        "end_close": _num(target_close),
        # Aliases kept for backward compatibility
        "session_open": _num(sess.get("open")),
        "session_close": _num(session_close),
        "target_open": _num(targ.get("open")),
        "target_close": _num(target_close),
        "return_pct": _num(label_return),
        "resolved": resolved,
        "actual_direction": actual_direction,
        "validation_basis": "close_to_close",
        "horizon_label": f"close {session_date} → close {target_str}",
    }


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
def root():
    """API root — UI is served by Next.js (`web/`)."""
    return jsonify({
        "service": "news-to-alpha-api",
        "health": "/healthz",
        "data_status": "/api/data-status",
        "ui": "Deploy the Next.js app in web/ (see docs/DEPLOY_UI.md)",
    })


@app.route("/healthz")
def healthz():
    """Liveness probe for Railway. Vercel proxies via web/app/api/healthz → this path."""
    return jsonify({
        "status": "ok",
        "predictions_csv": PREDICTIONS_CSV.exists(),
        "metrics_csv": EVAL_OVERALL_CSV.exists(),
        "tickers": len(TICKERS),
    })


@app.route("/api/data-status")
def api_data_status():
    """Freshness summary powering the header chip and status page."""
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

    # Freshness: predictions should cover through latest price session
    pipeline_cfg = _load_pipeline_cfg()
    horizon = int(pipeline_cfg.get("horizon", 1))
    last_session = last_trading_session().isoformat()
    behind = prediction_lag_sessions(latest_pred, latest_price)
    is_current = behind == 0 and latest_pred is not None
    expected_latest = latest_price

    # Last publish timestamp from bundle stamp
    last_published = None
    try:
        stamp_path = PROCESSED_DATA_DIR / "last_published.json"
        if stamp_path.exists():
            import json as _json
            last_published = _json.loads(stamp_path.read_text()).get("published_at")
    except Exception:
        pass

    # Resolved outcomes: most recent date where >= half the tickers have actual_binary filled
    latest_resolved = None
    try:
        if df is not None and not df.empty and "actual_binary" in df.columns:
            resolved_df = df.dropna(subset=["actual_binary"])
            if not resolved_df.empty:
                resolved_counts = resolved_df.groupby("prediction_date")["ticker"].count()
                total_counts = df.groupby("prediction_date")["ticker"].count()
                # Align indices — not every forecast date has resolved rows yet
                resolved_aligned = resolved_counts.reindex(total_counts.index, fill_value=0)
                majority = resolved_aligned[resolved_aligned >= (total_counts / 2)]
                if not majority.empty:
                    latest_resolved = str(majority.index.max())
    except Exception:
        pass

    # Market status based on current ET time vs 4 PM cutoff
    from datetime import datetime as _datetime, timezone as _tz
    import pytz as _pytz
    _ET = _pytz.timezone("US/Eastern")
    _now_et = _datetime.now(_tz.utc).astimezone(_ET)
    _today_et = _now_et.date()
    from src.utils.trading_calendar import _get_nyse  # reuse internal helper
    _nyse = _get_nyse()
    _today_is_trading = False
    try:
        if _nyse is not None:
            import pandas as _pd
            _today_is_trading = bool(_nyse.is_session(_pd.Timestamp(_today_et)))
        else:
            _today_is_trading = _today_et.weekday() < 5
    except Exception:
        _today_is_trading = _today_et.weekday() < 5

    if not _today_is_trading:
        market_status = "closed"
    elif _now_et.hour < 9 or (_now_et.hour == 9 and _now_et.minute < 30):
        market_status = "pre_market"
    elif _now_et.hour < 16:
        market_status = "open"
    else:
        market_status = "closed"

    # pending_reason for the latest forecast date
    pending_reason = "resolved"
    if latest_pred and (not latest_resolved or latest_pred > latest_resolved):
        if market_status == "open" or market_status == "pre_market":
            pending_reason = "awaiting_next_close"
        elif behind == 0:
            pending_reason = "awaiting_data_refresh"
        else:
            pending_reason = "awaiting_data_refresh"

    return _jsonify_safe({
        "today": today,
        "latest_prediction_date": latest_pred,
        "latest_price_date": latest_price,
        "latest_news_date": latest_news,
        "latest_resolved_prediction_date": latest_resolved,
        "prediction_rows": rows,
        "price_rows": price_count,
        "news_rows": news_count,
        "last_trading_session": last_session,
        "expected_latest_prediction_date": expected_latest,
        "trading_sessions_behind": behind,
        "is_current": is_current,
        "market_status": market_status,
        "pending_reason": pending_reason,
        "horizon": horizon,
        "last_published_at": last_published,
        "deploy_mode": "inference_only" if _INFERENCE_ONLY else "full",
        "train_config": {
            "encoder_model": pipeline_cfg.get("encoder_model") or ("finbert" if pipeline_cfg.get("use_finbert") else "minilm"),
            "conditional_ensemble": bool(pipeline_cfg.get("conditional_ensemble")),
            "min_move_pct": pipeline_cfg.get("min_move_pct"),
            "lstm_epochs": pipeline_cfg.get("lstm_epochs"),
        },
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

    price_ctx = _price_context(pred.ticker, pred.prediction_date)

    return jsonify({
        "ticker": pred.ticker,
        "company": TICKER_TO_COMPANY.get(pred.ticker, pred.ticker),
        "prediction_date": pred.prediction_date,
        "forecast_date": price_ctx.get("end_close_date") or price_ctx.get("target_date"),
        "model": pred.model,
        "proba": pred.proba,
        "binary": pred.binary,
        "confidence": pred.confidence,
        "actual_binary": pred.actual_binary,
        "hit": hit,
        "realized_return": realized_return,
        "top_headlines": pred.top_headlines,
        "per_model": pred.per_model,
        "price_context": price_ctx,
    })


@app.route("/api/last-resolved")
def api_last_resolved():
    """Return the most recent N predictions for a ticker that have a known actual.

    Powers the 'Yesterday vs Actual' card and the 7-dot history strip.
    """
    ticker = (request.args.get("ticker") or "").upper()
    model = (request.args.get("model") or "ensemble").lower()
    try:
        n = max(1, min(120, int(request.args.get("n") or "7")))
    except ValueError:
        n = 7
    if ticker not in TICKERS:
        return jsonify({"error": f"Unknown ticker: {ticker}"}), 404
    if model not in ALLOWED_MODELS:
        return jsonify({"error": f"Unknown model: {model}"}), 400

    df = _load_predictions()
    if df is None:
        return jsonify({"ticker": ticker, "rows": []})

    sub = df[df["ticker"] == ticker].copy()
    sub = sub[sub["actual_binary"].notna()].sort_values("prediction_date")
    if sub.empty:
        return jsonify({"ticker": ticker, "rows": []})

    sub = sub.tail(n)
    pred_binary = _model_pred_binary(sub, model)

    out: list[dict] = []
    for idx, (_, row) in enumerate(sub.iterrows()):
        proba_col = MODEL_PROBA_COL.get(model, "ensemble_pred_proba")
        proba = float(row[proba_col]) if proba_col in row and pd.notna(row.get(proba_col)) else None
        pred_bin = int(pred_binary.iloc[idx])
        actual = int(row["actual_binary"]) if pd.notna(row.get("actual_binary")) else None
        hit = int(pred_bin == actual) if actual is not None else None
        date_str = str(row["prediction_date"])
        ctx = _price_context(ticker, date_str)
        out.append({
            "date": date_str,
            "proba": proba,
            "pred_binary": pred_bin,
            "actual_binary": actual,
            "hit": hit,
            "return": ctx.get("return_pct"),
            "session_close": ctx.get("session_close"),
            "target_close": ctx.get("target_close"),
            "target_date": ctx.get("target_date"),
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
        preds_records = _df_records(sub, existing)

    prices = _recent_prices(ticker, days=(window if window else None))

    return _jsonify_safe({
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
    if _INFERENCE_ONLY:
        return jsonify({"error": "Training is disabled in inference-only mode."}), 403
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
    if _INFERENCE_ONLY:
        return jsonify({"error": "Model reset is disabled in inference-only mode."}), 403
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
    """Collect latest prices + news, run inference-only scoring, rebuild ensemble.

    When INFERENCE_ONLY=true (cloud mode): uses daily_update.py (no retrain).
    Local mode: runs full run_pipeline.py (includes retraining).

    Modes (local only):
      - ``quality`` (default): 3-seed LSTM retraining (~3 min).
      - ``fast``:              1-seed LSTM retraining (~1 min).
    """
    if _INFERENCE_ONLY:
        def run(job: JobSpec) -> None:
            cfg = _load_pipeline_cfg()
            cmd_args: list[str] = ["--lookback-days", str(int(days))]
            if not include_news:
                cmd_args.append("--skip-news")
            rc = jobs.python_script(job, "daily_update.py", *cmd_args)
            if rc != 0:
                raise RuntimeError("daily_update.py failed")
        return run

    # Full retrain path (local)
    seeds = ["42", "1337", "2024"] if str(mode).lower() != "fast" else ["42"]
    saved_cfg = _load_pipeline_cfg()
    horizon = str(saved_cfg.get("horizon", 1))
    tickers: list[str] = saved_cfg.get("tickers") or list(TICKERS)

    def run(job: JobSpec) -> None:
        cmd_args: list[str] = [
            "--tickers", *tickers,
            "--lookback-days", str(int(days)),
            "--seeds", *seeds,
            "--horizon", horizon,
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
    if _INFERENCE_ONLY:
        return jsonify({"error": "Full pipeline runs are disabled in inference-only mode. Use /api/data/refresh."}), 403
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
        title = r["title"] or ""
        sent = float(r["sentiment_score"]) if r["sentiment_score"] is not None else None
        rel = float(r["relevance_score"]) if r["relevance_score"] is not None else None
        out.append({
            "title": title,
            "headline": title,
            "source": r["source"] or "unknown",
            "published_at": r["published_at"],
            "content": (r["content"] or "")[:1200] if "content" in r.keys() else "",
            "sentiment_finnhub": sent,
            "finnhub_sentiment": sent,
            "relevance": rel,
            "relevance_score": rel,
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

_FEATURE_LABELS: dict[str, str] = {
    "financial_pred_proba": "Price P(UP)",
    "news_tfidf_pred_proba": "Keywords P(UP)",
    "news_embeddings_pred_proba": "FinBERT P(UP)",
    "lstm_confidence": "Price conviction",
    "tfidf_confidence": "Keyword conviction",
    "emb_confidence": "FinBERT conviction",
    "all_agree": "All models agree",
    "has_news": "Headlines present",
    "n_headlines": "Headline count",
    "spy_return_5d": "SPY 5-day return",
    "news_tfidf_x_has_news": "Keywords × headlines",
    "news_emb_x_has_news": "FinBERT × headlines",
    "lstm_x_agree": "Price × agreement",
}

_FEATURE_HINTS: dict[str, str] = {
    "financial_pred_proba": "LSTM output from 60-day price + technicals + VIX.",
    "news_tfidf_pred_proba": "TF-IDF headline model.",
    "news_embeddings_pred_proba": "FinBERT headline model.",
    "spy_return_5d": "Broad market regime — SPY 5-day % change.",
    "has_news": "1 when cutoff-aligned headlines exist before 4 PM ET.",
    "n_headlines": "Headlines counted for that session.",
    "all_agree": "1 when price, keywords, and FinBERT agree on direction.",
    "news_tfidf_x_has_news": "Keyword score gated by headline presence.",
    "news_emb_x_has_news": "FinBERT score gated by headline presence.",
    "lstm_x_agree": "Price score boosted when all models agree.",
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


def _meta_payload_for_explanation(row: pd.Series, full_payload: dict) -> dict[str, Any]:
    """Select the correct meta-model for counterfactual explanation on one row."""
    if full_payload.get("conditional"):
        has_news = int(row.get("has_news", 0) or 0) == 1
        if has_news:
            return {
                "meta": full_payload.get("has_news_model"),
                "temperature": float(full_payload.get("has_news_temperature", 1.0)),
                "importances": full_payload.get("importances") or [],
                "conditional": True,
                "route": "has_news",
            }
        return {
            "meta": full_payload.get("no_news_model"),
            "temperature": float(full_payload.get("no_news_temperature", 1.0)),
            "importances": full_payload.get("importances") or [],
            "conditional": True,
            "route": "no_news",
        }
    return {
        "meta": full_payload.get("meta"),
        "temperature": float(full_payload.get("temperature", 1.0)),
        "importances": full_payload.get("importances") or [],
        "conditional": False,
        "route": "unified",
    }


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
    ensemble_route: str | None = None
    if META_MODEL_PATH.exists():
        try:
            payload = joblib.load(META_MODEL_PATH)
            features = payload.get("features", [])
            importances = payload.get("importances", []) or []
            meta_for_row = _meta_payload_for_explanation(row, payload)
            temperature = float(meta_for_row.get("temperature", 1.0))
            ensemble_route = meta_for_row.get("route")
            if payload.get("conditional"):
                importances = meta_for_row.get("importances") or importances
        except Exception:
            features, importances = [], []
            meta_for_row = {"meta": None, "temperature": 1.0, "importances": []}
    else:
        meta_for_row = {"meta": None, "temperature": 1.0, "importances": []}

    imp_map = {k: v for k, v in importances}
    scales = _compute_feature_scales(df, features)

    # Legacy linear contributions (kept for compatibility)
    contributions: list[dict] = []
    for name in features:
        if name not in row_df.columns or pd.isna(row[name]):
            continue
        value = float(row[name])
        imp = float(imp_map.get(name, 0.0))
        if imp <= 1e-8:
            continue
        sc = scales.get(name, {"baseline": 0.0, "scale": 1.0})
        baseline = sc["baseline"]
        scale = sc["scale"] if sc["scale"] > 0 else 1.0
        contribution = imp * (value - baseline) / scale
        if abs(contribution) < 1e-8:
            direction = "neutral"
        elif contribution > 0:
            direction = "up"
        else:
            direction = "down"
        contributions.append({
            "feature": name,
            "label": _FEATURE_LABELS.get(name, name.replace("_", " ").title()),
            "hint": _FEATURE_HINTS.get(name, ""),
            "value": value,
            "importance": imp,
            "baseline": baseline,
            "contribution": contribution,
            "direction": direction,
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

    explanation: dict[str, Any] = {}
    try:
        from src.ml.ensemble_explain import explain_ensemble_row

        if META_MODEL_PATH.exists() and meta_for_row.get("meta") is not None:
            explanation = explain_ensemble_row(row, meta_for_row)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("ensemble explain failed: %s", exc)
        explanation = {}

    has_news = int(row.get("has_news", 0) or 0) == 1

    from scripts.build_ensemble import META_FEATURES
    from src.features.lstm_snapshot import get_lstm_snapshot

    meta_features: list[dict[str, Any]] = []
    for feat in META_FEATURES:
        if feat in row.index and pd.notna(row[feat]):
            val = float(row[feat])
        else:
            val = 0.0
        meta_features.append({
            "key": feat,
            "label": _FEATURE_LABELS.get(feat, feat.replace("_", " ")),
            "hint": _FEATURE_HINTS.get(feat, ""),
            "value": val,
            "importance": float(imp_map.get(feat, 0.0)),
        })

    lstm_context = get_lstm_snapshot(ticker, str(row["prediction_date"]))

    return _jsonify_safe({
        "ticker": ticker,
        "date": str(row["prediction_date"]),
        "per_model": per_model,
        "ensemble_proba": float(row.get("ensemble_pred_proba", 0.5)) if "ensemble_pred_proba" in row_df.columns else None,
        "ensemble_confidence": float(row.get("ensemble_confidence", 0.0)) if "ensemble_confidence" in row_df.columns else None,
        "has_news": has_news,
        "ensemble_route": ensemble_route,
        "meta_features": meta_features,
        "lstm_context": lstm_context,
        "contributions": contributions[:8],
        "features": contributions[:8],
        "explanation": explanation,
        "temperature": temperature,
    })


@app.route("/api/accuracy-trace")
def api_accuracy_trace():
    ticker = (request.args.get("ticker") or "").upper()
    model = (request.args.get("model") or "ensemble").lower()
    try:
        window = max(5, int(request.args.get("window") or "30"))
    except ValueError:
        window = 30
    if model not in ALLOWED_MODELS:
        return jsonify({"error": f"Unknown model: {model}"}), 400

    df = _load_predictions()
    if df is None:
        return jsonify({"ticker": ticker, "model": model, "window": window, "series": []})

    sub = df[df["ticker"] == ticker].copy() if ticker else df.copy()
    sub = sub[sub["actual_binary"].notna()].sort_values("prediction_date")
    if sub.empty:
        return jsonify({"ticker": ticker, "model": model, "window": window, "series": []})

    pred_binary = _model_pred_binary(sub, model)
    sub["correct"] = (pred_binary == sub["actual_binary"].astype(int)).astype(int)
    sub["rolling_acc"] = sub["correct"].rolling(window=window, min_periods=5).mean()

    series = [
        {"date": d, "accuracy": (float(a) if pd.notna(a) else None)}
        for d, a in zip(sub["prediction_date"], sub["rolling_acc"])
    ]
    return _jsonify_safe({"ticker": ticker, "model": model, "window": window, "series": series})


def _markets_price_index(window: int) -> list[dict[str, Any]]:
    """Equal-weight normalized price index across TICKERS (100 at window start)."""
    conn = _get_db()
    try:
        date_rows = conn.execute(
            """SELECT DISTINCT date FROM prices ORDER BY date DESC LIMIT ?""",
            (window,),
        ).fetchall()
    finally:
        conn.close()

    if not date_rows:
        return []

    dates = sorted(str(r["date"]) for r in date_rows)
    by_date: dict[str, list[float]] = {d: [] for d in dates}

    for ticker in TICKERS:
        conn = _get_db()
        try:
            placeholders = ",".join(["?"] * len(dates))
            rows = conn.execute(
                f"""SELECT date, close FROM prices
                    WHERE ticker = ? AND date IN ({placeholders})
                    ORDER BY date ASC""",
                (ticker, *dates),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            continue
        base = float(rows[0]["close"])
        if base <= 0:
            continue
        for r in rows:
            d = str(r["date"])
            by_date.setdefault(d, []).append(float(r["close"]) / base * 100.0)

    out: list[dict[str, Any]] = []
    for d in dates:
        vals = by_date.get(d, [])
        if vals:
            out.append({"date": d, "index": float(sum(vals) / len(vals))})
    return out


@app.route("/api/markets-overview")
def api_markets_overview():
    """Aggregate price index + daily ensemble accuracy for all tickers."""
    try:
        window = max(7, min(90, int(request.args.get("window") or "30")))
    except ValueError:
        window = 30

    df = _load_predictions()
    if df is None:
        return _jsonify_safe({
            "window": window,
            "price_index": [],
            "accuracy_series": [],
            "summary": {"n": 0, "hits": 0, "accuracy": None},
        })

    sub = df[df["actual_binary"].notna()].copy()
    sub["hit"] = (sub["ensemble_pred_binary"] == sub["actual_binary"]).astype(int)

    daily = (
        sub.groupby("prediction_date")
        .agg(hits=("hit", "sum"), n=("hit", "count"))
        .reset_index()
        .sort_values("prediction_date")
    )
    daily["accuracy"] = daily["hits"] / daily["n"]
    daily = daily.tail(window)

    accuracy_series = [
        {"date": str(r["prediction_date"]), "accuracy": float(r["accuracy"])}
        for _, r in daily.iterrows()
    ]

    # Summary over the same calendar dates as the chart window
    if daily.empty:
        summary = {"n": 0, "hits": 0, "accuracy": None}
    else:
        window_dates = set(daily["prediction_date"].astype(str))
        trim = sub[sub["prediction_date"].astype(str).isin(window_dates)]
        n = int(len(trim))
        hits = int(trim["hit"].sum())
        summary = {"n": n, "hits": hits, "accuracy": (hits / n) if n else None}

    return _jsonify_safe({
        "window": window,
        "price_index": _markets_price_index(window),
        "accuracy_series": accuracy_series,
        "summary": summary,
    })


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
    model = (request.args.get("model") or "ensemble").lower()
    window_raw = (request.args.get("window") or "30").lower()
    if model not in ALLOWED_MODELS:
        return jsonify({"error": f"Unknown model: {model}"}), 400
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
    pred_binary = _model_pred_binary(window_trim, model)
    window_trim["hit"] = (pred_binary == window_trim["actual_binary"].astype(int)).astype(int)
    proba_col = MODEL_PROBA_COL.get(model, "ensemble_pred_proba")

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
    for i, (_, r) in enumerate(window_trim.iterrows()):
        key = (r["ticker"], str(r["prediction_date"]))
        rows.append({
            "date": str(r["prediction_date"]),
            "ticker": r["ticker"],
            "pred_binary": int(pred_binary.iloc[i]),
            "actual_binary": int(r["actual_binary"]) if pd.notna(r["actual_binary"]) else None,
            "hit": int(window_trim["hit"].iloc[i]),
            "proba": float(r[proba_col]) if proba_col in r.index and pd.notna(r.get(proba_col)) else None,
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
