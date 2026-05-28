#!/usr/bin/env python3
"""Inference-only scoring: load saved model artifacts and write prediction CSVs.

Run after collect_prices / collect_news / generate_labels when you want to
advance prediction dates WITHOUT retraining. Reads existing .pt / .joblib
files and writes the same CSVs that the train_* scripts produce.

Outputs (same paths as training)
---------------------------------
  data/processed/price_predictions.csv
  data/processed/news_tfidf_predictions.csv
  data/processed/news_embeddings_predictions.csv

Usage
-----
  python scripts/score_models.py                  # all three models
  python scripts/score_models.py --skip-lstm
  python scripts/score_models.py --dry-run        # report what would run, exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    DATABASE_PATH,
    TICKERS,
)
from src.utils.pipeline_config import load_or_default  # noqa: E402


# ---------------------------------------------------------------------------
# LSTM inference
# ---------------------------------------------------------------------------

def _score_lstm(horizon: int, dry_run: bool) -> bool:
    model_path = MODELS_DIR / "lstm_model.pt"

    if not model_path.exists():
        print(f"[score_lstm] SKIP – model not found: {model_path}")
        return False

    if dry_run:
        print(f"[score_lstm] DRY-RUN – would append live rows, horizon={horizon}")
        return True

    from src.ml.lstm_live_export import append_live_lstm_predictions  # noqa: E402

    cfg = load_or_default()
    tickers = cfg.get("tickers") or list(TICKERS)
    n = append_live_lstm_predictions(tickers, horizon=horizon, model_path=model_path)
    print(f"[score_lstm] Appended {n} live rows to price_predictions.csv")
    return n > 0


# ---------------------------------------------------------------------------
# TF-IDF NLP inference
# ---------------------------------------------------------------------------

def _score_tfidf(horizon: int, dry_run: bool) -> bool:
    model_path = MODELS_DIR / "nlp_baseline.joblib"
    out_csv = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"

    if not model_path.exists():
        print(f"[score_tfidf] SKIP – model not found: {model_path}")
        return False

    print(f"[score_tfidf] Loading model from {model_path}")
    if dry_run:
        print("[score_tfidf] DRY-RUN – would score all news-bearing days")
        return True

    from src.models.news_pipeline import build_dataset  # noqa: E402
    from scripts.train_nlp import export_predictions, load_tfidf_model  # noqa: E402

    model = load_tfidf_model(model_path)
    df = build_dataset(DATABASE_PATH, drop_rows_without_news=True, horizon=horizon)
    if df.empty:
        print("[score_tfidf] No news-bearing rows – skipping.")
        return False

    out_df = export_predictions(model, "daily", df)
    col_order = [
        "ticker", "prediction_date", "split", "model_name",
        "news_pred_proba", "news_pred_binary", "news_confidence",
        "top_headlines", "actual_binary", "model_version",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df[col_order].to_csv(out_csv, index=False)
    print(f"[score_tfidf] Saved {len(out_df)} rows → {out_csv}")

    # Append live-day predictions (no labels yet) via news_live_export
    from src.ml.news_live_export import append_live_tfidf_predictions  # noqa: E402
    n_live = append_live_tfidf_predictions(model_path=model_path, out_csv=out_csv)
    if n_live:
        print(f"[score_tfidf] Appended {n_live} live rows → {out_csv}")
    return True


# ---------------------------------------------------------------------------
# Embeddings inference
# ---------------------------------------------------------------------------

def _score_embeddings(horizon: int, dry_run: bool) -> bool:
    model_path = MODELS_DIR / "news_embeddings.joblib"
    out_csv = PROCESSED_DATA_DIR / "news_embeddings_predictions.csv"

    if not model_path.exists():
        print(f"[score_embeddings] SKIP – model not found: {model_path}")
        return False

    print(f"[score_embeddings] Loading model from {model_path}")
    if dry_run:
        print("[score_embeddings] DRY-RUN – would score all news-bearing days")
        return True

    from src.models.news_pipeline import build_dataset  # noqa: E402
    from scripts.train_news_embeddings import (  # noqa: E402
        export_predictions,
        load_embedding_model,
    )

    model = load_embedding_model(model_path)
    df = build_dataset(DATABASE_PATH, drop_rows_without_news=True, horizon=horizon)
    if df.empty:
        print("[score_embeddings] No news-bearing rows – skipping.")
        return False

    out_df = export_predictions(model, "daily", df)
    col_order = [
        "ticker", "prediction_date", "split", "model_name",
        "news_pred_proba", "news_pred_binary", "news_confidence",
        "top_headlines", "actual_binary", "model_version",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df[col_order].to_csv(out_csv, index=False)
    print(f"[score_embeddings] Saved {len(out_df)} rows → {out_csv}")

    # Append live-day predictions (no labels yet) via news_live_export
    from src.ml.news_live_export import append_live_embedding_predictions  # noqa: E402
    n_live = append_live_embedding_predictions(model_path=model_path, out_csv=out_csv)
    if n_live:
        print(f"[score_embeddings] Appended {n_live} live rows → {out_csv}")
    return True


# ---------------------------------------------------------------------------
# Backfill actual_binary from labels onto live prediction rows
# ---------------------------------------------------------------------------

def backfill_outcomes(dry_run: bool = False) -> int:
    """Join labels from the DB onto live rows in price and ensemble CSVs.

    Returns total number of rows updated across both CSVs.
    """
    import sqlite3
    conn = sqlite3.connect(str(DATABASE_PATH))
    try:
        labels = pd.read_sql_query(
            "SELECT ticker, date AS prediction_date, label_binary AS actual_binary FROM labels",
            conn,
        )
    finally:
        conn.close()

    if labels.empty:
        return 0

    labels["prediction_date"] = labels["prediction_date"].astype(str)
    label_map: dict[tuple, int] = {
        (r["ticker"], r["prediction_date"]): int(r["actual_binary"])
        for _, r in labels.dropna(subset=["actual_binary"]).iterrows()
    }

    total = 0
    csv_paths = [
        PROCESSED_DATA_DIR / "price_predictions.csv",
        PROCESSED_DATA_DIR / "news_tfidf_predictions.csv",
        PROCESSED_DATA_DIR / "news_embeddings_predictions.csv",
        PROCESSED_DATA_DIR / "final_ensemble_predictions.csv",
    ]
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["prediction_date"] = df["prediction_date"].astype(str)
        null_mask = df["actual_binary"].isna()
        if not null_mask.any():
            continue

        def _fill(row):
            if pd.isna(row["actual_binary"]):
                return label_map.get((row["ticker"], row["prediction_date"]))
            return row["actual_binary"]

        updated = df[null_mask].apply(_fill, axis=1)
        filled = updated.notna().sum()
        if filled == 0:
            continue
        if dry_run:
            print(f"[backfill_outcomes] DRY-RUN – would fill {filled} rows in {csv_path.name}")
            total += filled
            continue
        df.loc[null_mask, "actual_binary"] = updated
        if (
            "hit" in df.columns
            and "ensemble_pred_binary" in df.columns
            and csv_path.name == "final_ensemble_predictions.csv"
        ):
            resolved = df["actual_binary"].notna()
            df.loc[resolved, "hit"] = (
                df.loc[resolved, "ensemble_pred_binary"].astype(int)
                == df.loc[resolved, "actual_binary"].astype(int)
            ).astype(int)
        df.to_csv(csv_path, index=False)
        print(f"[backfill_outcomes] Filled {filled} actual_binary values in {csv_path.name}")
        total += filled

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only model scoring")
    parser.add_argument("--horizon", type=int, default=None, choices=[1, 3],
                        help="Prediction horizon (default: from pipeline_config.json)")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip backfilling actual_binary on live rows.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would run without executing.")
    args = parser.parse_args()

    cfg = load_or_default()
    horizon = args.horizon or int(cfg.get("horizon", 1))
    print(f"[score_models] horizon={horizon}  dry_run={args.dry_run}")

    if not args.skip_lstm:
        _score_lstm(horizon, args.dry_run)
    if not args.skip_tfidf:
        _score_tfidf(horizon, args.dry_run)
    if not args.skip_embeddings:
        _score_embeddings(horizon, args.dry_run)
    if not args.skip_backfill:
        n = backfill_outcomes(dry_run=args.dry_run)
        print(f"[score_models] backfill_outcomes: {n} rows updated")

    print("[score_models] Done.")


if __name__ == "__main__":
    main()
