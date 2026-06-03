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
  data/processed/volatility_predictions.csv

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

def _score_tfidf(horizon: int, dry_run: bool, incremental: bool = False) -> bool:
    model_path = MODELS_DIR / "nlp_baseline.joblib"
    out_csv = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"

    if not model_path.exists():
        print(f"[score_tfidf] SKIP – model not found: {model_path}")
        return False

    if dry_run:
        mode = "live rows only" if incremental else "all news-bearing days"
        print(f"[score_tfidf] DRY-RUN – would score {mode}")
        return True

    from src.ml.news_live_export import append_live_tfidf_predictions  # noqa: E402

    if incremental:
        if not out_csv.exists():
            print(
                "[score_tfidf] Incremental – no historical CSV yet; "
                "scoring live rows only (publish bundle to seed history)"
            )
        else:
            print("[score_tfidf] Incremental – skipping full historical rescore")
        print(f"[score_tfidf] Loading model from {model_path}")
        n_live = append_live_tfidf_predictions(model_path=model_path, out_csv=out_csv)
        print(f"[score_tfidf] Appended {n_live} live rows → {out_csv}")
        return True

    print(f"[score_tfidf] Loading model from {model_path}")
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

    n_live = append_live_tfidf_predictions(model_path=model_path, out_csv=out_csv)
    if n_live:
        print(f"[score_tfidf] Appended {n_live} live rows → {out_csv}")
    return True


# ---------------------------------------------------------------------------
# Embeddings inference
# ---------------------------------------------------------------------------

def _score_embeddings(horizon: int, dry_run: bool, incremental: bool = False) -> bool:
    model_path = MODELS_DIR / "news_embeddings.joblib"
    out_csv = PROCESSED_DATA_DIR / "news_embeddings_predictions.csv"

    if not model_path.exists():
        print(f"[score_embeddings] SKIP – model not found: {model_path}")
        return False

    if dry_run:
        mode = "live rows only" if incremental else "all news-bearing days"
        print(f"[score_embeddings] DRY-RUN – would score {mode}")
        return True

    from src.ml.news_live_export import append_live_embedding_predictions  # noqa: E402

    if incremental:
        if not out_csv.exists():
            print(
                "[score_embeddings] Incremental – no historical CSV yet; "
                "scoring live rows only (publish bundle to seed history)"
            )
        else:
            print("[score_embeddings] Incremental – skipping full FinBERT historical rescore")
        print(f"[score_embeddings] Loading model from {model_path}")
        n_live = append_live_embedding_predictions(model_path=model_path, out_csv=out_csv)
        print(f"[score_embeddings] Appended {n_live} live rows → {out_csv}")
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

    print(f"[score_embeddings] Full rescore of {len(df)} news rows (slow on CPU — use --incremental for daily runs)")
    out_df = export_predictions(model, "daily", df)
    col_order = [
        "ticker", "prediction_date", "split", "model_name",
        "news_pred_proba", "news_pred_binary", "news_confidence",
        "top_headlines", "actual_binary", "model_version",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df[col_order].to_csv(out_csv, index=False)
    print(f"[score_embeddings] Saved {len(out_df)} rows → {out_csv}")

    n_live = append_live_embedding_predictions(model_path=model_path, out_csv=out_csv)
    if n_live:
        print(f"[score_embeddings] Appended {n_live} live rows → {out_csv}")
    return True


# ---------------------------------------------------------------------------
# Volatility inference
# ---------------------------------------------------------------------------

def _score_volatility(horizon: int, dry_run: bool) -> bool:
    model_path = MODELS_DIR / "volatility_model.joblib"
    out_csv = PROCESSED_DATA_DIR / "volatility_predictions.csv"

    if not model_path.exists():
        print(f"[score_volatility] SKIP – model not found: {model_path}")
        return False

    if dry_run:
        print(f"[score_volatility] DRY-RUN – would append live rows, horizon={horizon}")
        return True

    from src.ml.volatility_live_export import append_live_volatility_predictions  # noqa: E402

    cfg = load_or_default()
    tickers = cfg.get("tickers") or list(TICKERS)
    try:
        n = append_live_volatility_predictions(tickers=tickers, horizon=horizon)
    except Exception as exc:
        print(
            f"[score_volatility] SKIP – live scoring failed: {exc}\n"
            "  Daily update continues; direction/news scores are unaffected.\n"
            "  Retrain volatility and republish, or fix sklearn version mismatch (see docs/DATA.md)."
        )
        return False
    print(f"[score_volatility] Appended {n} live rows → {out_csv}")
    return n > 0


# ---------------------------------------------------------------------------
# Backfill actual_binary from labels onto live prediction rows
# ---------------------------------------------------------------------------

def backfill_outcomes(dry_run: bool = False) -> int:
    """Join labels from the DB onto live rows in prediction CSVs.

    Each CSV declares its own outcome column (direction vs volatility).
    """
    import sqlite3
    conn = sqlite3.connect(str(DATABASE_PATH))
    try:
        labels = pd.read_sql_query(
            "SELECT ticker, date AS prediction_date, label_binary, label_return FROM labels",
            conn,
        )
    finally:
        conn.close()

    if labels.empty:
        return 0

    labels["prediction_date"] = labels["prediction_date"].astype(str)
    label_map: dict[tuple, int] = {
        (r["ticker"], r["prediction_date"]): int(r["label_binary"])
        for _, r in labels.dropna(subset=["label_binary"]).iterrows()
    }
    abs_return_map: dict[tuple, float] = {
        (r["ticker"], r["prediction_date"]): abs(float(r["label_return"])) * 100.0
        for _, r in labels.dropna(subset=["label_return"]).iterrows()
    }

    # (filename, outcome column, lookup map)
    specs: list[tuple[str, str, dict]] = [
        ("price_predictions.csv", "actual_binary", label_map),
        ("news_tfidf_predictions.csv", "actual_binary", label_map),
        ("news_embeddings_predictions.csv", "actual_binary", label_map),
        ("final_ensemble_predictions.csv", "actual_binary", label_map),
        ("volatility_predictions.csv", "actual_abs_return_pct", abs_return_map),
    ]

    total = 0
    for filename, outcome_col, value_map in specs:
        if not value_map:
            continue
        csv_path = PROCESSED_DATA_DIR / filename
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if outcome_col not in df.columns:
            continue
        df["prediction_date"] = df["prediction_date"].astype(str)
        null_mask = df[outcome_col].isna()
        if not null_mask.any():
            continue

        keys = list(zip(df.loc[null_mask, "ticker"], df.loc[null_mask, "prediction_date"]))
        filled_vals = [value_map.get(k) for k in keys]
        filled = sum(v is not None for v in filled_vals)
        if filled == 0:
            continue
        if dry_run:
            print(
                f"[backfill_outcomes] DRY-RUN – would fill {filled} "
                f"{outcome_col} in {filename}"
            )
            total += filled
            continue
        for idx, val in zip(df.index[null_mask], filled_vals):
            if val is not None:
                df.at[idx, outcome_col] = val
        if (
            outcome_col == "actual_binary"
            and "hit" in df.columns
            and "ensemble_pred_binary" in df.columns
            and filename == "final_ensemble_predictions.csv"
        ):
            resolved = df["actual_binary"].notna()
            df.loc[resolved, "hit"] = (
                df.loc[resolved, "ensemble_pred_binary"].astype(int)
                == df.loc[resolved, "actual_binary"].astype(int)
            ).astype(int)
        df.to_csv(csv_path, index=False)
        print(f"[backfill_outcomes] Filled {filled} {outcome_col} values in {filename}")
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
    parser.add_argument("--skip-volatility", action="store_true")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip backfilling actual_binary on live rows.")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Daily mode: score only new live rows for NLP/embeddings (skip full FinBERT rescore).",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would run without executing.")
    args = parser.parse_args()

    cfg = load_or_default()
    horizon = args.horizon or int(cfg.get("horizon", 1))
    print(f"[score_models] horizon={horizon}  incremental={args.incremental}  dry_run={args.dry_run}")

    if not args.skip_lstm:
        _score_lstm(horizon, args.dry_run)
    if not args.skip_tfidf:
        _score_tfidf(horizon, args.dry_run, incremental=args.incremental)
    if not args.skip_embeddings:
        _score_embeddings(horizon, args.dry_run, incremental=args.incremental)
    if not args.skip_volatility:
        _score_volatility(horizon, args.dry_run)
    if not args.skip_backfill:
        n = backfill_outcomes(dry_run=args.dry_run)
        print(f"[score_models] backfill_outcomes: {n} rows updated")

    print("[score_models] Done.")


if __name__ == "__main__":
    main()
