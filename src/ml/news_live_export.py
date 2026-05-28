"""Score news models for prediction dates that have headlines but no label yet.

The TF-IDF and embedding scorers in score_models.py call build_dataset(), which
requires a join against the labels table. This means live dates (split=live) never
get news predictions — the UI shows "No headlines" even when 30+ articles exist.

This module mirrors the LSTM live-export pattern: it finds live prediction dates in
price_predictions.csv that have DB headlines, runs the saved news models on them,
and appends rows with split="live" / actual_binary=NaN to the news prediction CSVs.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATABASE_PATH, MODELS_DIR, PROCESSED_DATA_DIR

MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")

_TFIDF_CSV = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"
_EMB_CSV = PROCESSED_DATA_DIR / "news_embeddings_predictions.csv"
_PRICE_CSV = PROCESSED_DATA_DIR / "price_predictions.csv"


def _live_dates_needing_news(price_csv: Path, news_csv: Path) -> set[tuple[str, str]]:
    """Return (ticker, date) pairs needing news scores for live price sessions."""
    pairs: set[tuple[str, str]] = set()
    if price_csv.exists():
        price = pd.read_csv(price_csv, usecols=["ticker", "prediction_date", "split"])
        live = price[price["split"] == "live"][["ticker", "prediction_date"]]
        if not live.empty:
            pairs |= set(zip(live["ticker"], live["prediction_date"].astype(str)))

    if news_csv.exists():
        news = pd.read_csv(news_csv, usecols=["ticker", "prediction_date"])
        existing = set(zip(news["ticker"], news["prediction_date"].astype(str)))
        return pairs - existing
    return pairs


def _load_news_for_pairs(
    db_path: Path,
    pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    """Load cutoff-aligned news from DB restricted to the given (ticker, date) pairs."""
    from src.models.news_pipeline import _load_news_aligned

    if not db_path.exists() or not pairs:
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    try:
        news_all = _load_news_aligned(conn)
    finally:
        conn.close()

    if news_all.empty:
        return pd.DataFrame()

    news_all = news_all.rename(columns={"label_date": "prediction_date"})
    news_all["prediction_date"] = news_all["prediction_date"].astype(str)
    mask = news_all.apply(
        lambda r: (r["ticker"], r["prediction_date"]) in pairs, axis=1
    )
    return news_all[mask].reset_index(drop=True)


def _append_live_rows(out_df: pd.DataFrame, csv_path: Path, model_name: str) -> int:
    """Upsert live news rows into a news prediction CSV. Returns number of rows added."""
    col_order = [
        "ticker", "prediction_date", "split", "model_name",
        "news_pred_proba", "news_pred_binary", "news_confidence",
        "top_headlines", "actual_binary", "model_version",
    ]
    out_df = out_df.reindex(columns=col_order)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing["prediction_date"] = existing["prediction_date"].astype(str)
        out_df["prediction_date"] = out_df["prediction_date"].astype(str)
        # Remove any overlapping rows that we're about to re-score
        keys = set(zip(out_df["ticker"], out_df["prediction_date"]))
        existing = existing[
            ~existing.apply(
                lambda r: (r["ticker"], r["prediction_date"]) in keys, axis=1
            )
        ]
        combined = pd.concat([existing, out_df], ignore_index=True)
    else:
        combined = out_df

    combined = combined.sort_values(["ticker", "prediction_date"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return len(out_df)


def append_live_tfidf_predictions(
    db_path: Path | None = None,
    price_csv: Path | None = None,
    out_csv: Path | None = None,
    model_path: Path | None = None,
) -> int:
    """Score live dates via TF-IDF model and append to news_tfidf_predictions.csv."""
    db_path = db_path or DATABASE_PATH
    price_csv = price_csv or _PRICE_CSV
    out_csv = out_csv or _TFIDF_CSV
    model_path = model_path or (MODELS_DIR / "nlp_baseline.joblib")

    if not model_path.exists():
        return 0

    pairs = _live_dates_needing_news(price_csv, out_csv)
    if not pairs:
        return 0

    df = _load_news_for_pairs(db_path, pairs)
    df = df[df["n_headlines"] > 0].reset_index(drop=True)
    if df.empty:
        return 0

    from scripts.train_nlp import load_tfidf_model

    model = load_tfidf_model(model_path)
    proba = model.predict_proba_positive(df)
    binary = (proba >= 0.5).astype(int)
    conf = np.abs(proba - 0.5) * 2.0

    out = pd.DataFrame({
        "ticker": df["ticker"].values,
        "prediction_date": df["prediction_date"].values,
        "split": "live",
        "model_name": "news_tfidf",
        "news_pred_proba": proba,
        "news_pred_binary": binary,
        "news_confidence": conf,
        "top_headlines": df["top_headlines"].values,
        "actual_binary": np.nan,
        "model_version": MODEL_VERSION,
    })

    return _append_live_rows(out, out_csv, "news_tfidf")


def append_live_embedding_predictions(
    db_path: Path | None = None,
    price_csv: Path | None = None,
    out_csv: Path | None = None,
    model_path: Path | None = None,
) -> int:
    """Score live dates via embedding model and append to news_embeddings_predictions.csv."""
    db_path = db_path or DATABASE_PATH
    price_csv = price_csv or _PRICE_CSV
    out_csv = out_csv or _EMB_CSV
    model_path = model_path or (MODELS_DIR / "news_embeddings.joblib")

    if not model_path.exists():
        return 0

    pairs = _live_dates_needing_news(price_csv, out_csv)
    if not pairs:
        return 0

    df = _load_news_for_pairs(db_path, pairs)
    df = df[df["n_headlines"] > 0].reset_index(drop=True)
    if df.empty:
        return 0

    from scripts.train_news_embeddings import load_embedding_model

    model = load_embedding_model(model_path)
    proba = model.predict_proba_positive(df)
    binary = (proba >= 0.5).astype(int)
    conf = np.abs(proba - 0.5) * 2.0

    out = pd.DataFrame({
        "ticker": df["ticker"].values,
        "prediction_date": df["prediction_date"].values,
        "split": "live",
        "model_name": "news_embeddings",
        "news_pred_proba": proba,
        "news_pred_binary": binary,
        "news_confidence": conf,
        "top_headlines": df["top_headlines"].values,
        "actual_binary": np.nan,
        "model_version": MODEL_VERSION,
    })

    return _append_live_rows(out, out_csv, "news_embeddings")
