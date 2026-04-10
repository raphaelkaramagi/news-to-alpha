#!/usr/bin/env python3
"""
Build the final ensemble predictions from the aligned eval dataset.

Locked formula (docs/session_2_contract.md):
    ensemble_pred_proba = 0.45 * financial_pred_proba
                        + 0.25 * news_tfidf_pred_proba
                        + 0.30 * news_embeddings_pred_proba
    ensemble_pred_binary  = 1 if ensemble_pred_proba >= 0.5 else 0
    ensemble_confidence   = abs(ensemble_pred_proba - 0.5) * 2

Input:   data/processed/eval_dataset.csv      (built by scripts/build_eval_dataset.py)
Output:  data/processed/final_ensemble_predictions.csv

Usage
-----
    python scripts/build_ensemble.py
    python scripts/build_ensemble.py --input data/processed/eval_dataset.csv
    python scripts/build_ensemble.py --output data/processed/final_ensemble_predictions.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if not (_PROJECT_ROOT / "src").exists():
    _PROJECT_ROOT = _SCRIPT_DIR

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.config import PROCESSED_DATA_DIR  # noqa: E402
except ModuleNotFoundError:
    PROCESSED_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

# ── Ensemble formula weights (locked in session_2_contract.md) ────────────────
W_FINANCIAL       = 0.45
W_NEWS_TFIDF      = 0.25
W_NEWS_EMBEDDINGS = 0.30

MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")

# ── Required input columns ────────────────────────────────────────────────────
REQUIRED_COLS = [
    "ticker",
    "prediction_date",
    "financial_pred_proba",
    "news_tfidf_pred_proba",
    "news_embeddings_pred_proba",
    "actual_binary",
]

# ── Required output columns (contract order) ──────────────────────────────────
OUTPUT_COLS = [
    "ticker",
    "prediction_date",
    "split",
    "financial_pred_proba",
    "news_tfidf_pred_proba",
    "news_embeddings_pred_proba",
    "ensemble_pred_proba",
    "ensemble_pred_binary",
    "ensemble_confidence",
    "top_headlines",
    "actual_binary",
    "model_version",
]


def load_eval_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"eval_dataset.csv not found at {path}\n"
            "Run scripts/build_eval_dataset.py first."
        )
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"eval_dataset.csv is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def compute_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the locked weighted-average formula and derive binary + confidence."""
    df = df.copy()

    df["ensemble_pred_proba"] = (
        W_FINANCIAL       * df["financial_pred_proba"]
        + W_NEWS_TFIDF      * df["news_tfidf_pred_proba"]
        + W_NEWS_EMBEDDINGS * df["news_embeddings_pred_proba"]
    )

    df["ensemble_pred_binary"] = (df["ensemble_pred_proba"] >= 0.5).astype(int)
    df["ensemble_confidence"]  = np.abs(df["ensemble_pred_proba"] - 0.5) * 2
    df["model_version"]        = MODEL_VERSION

    # top_headlines — use embeddings column if present, else tfidf, else empty
    if "top_headlines" not in df.columns:
        if "news_embeddings_top_headlines" in df.columns:
            df["top_headlines"] = df["news_embeddings_top_headlines"]
        elif "news_tfidf_top_headlines" in df.columns:
            df["top_headlines"] = df["news_tfidf_top_headlines"]
        else:
            df["top_headlines"] = "[]"

    # split — use price_split as canonical if present (per contract)
    if "split" not in df.columns:
        if "price_split" in df.columns:
            df["split"] = df["price_split"]
        else:
            df["split"] = "unknown"

    # Keep only output columns that exist (graceful if optional cols absent)
    out_cols = [c for c in OUTPUT_COLS if c in df.columns]
    return df[out_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final ensemble predictions")
    parser.add_argument(
        "--input", default=None,
        help=f"Path to eval_dataset.csv (default: {PROCESSED_DATA_DIR / 'eval_dataset.csv'})",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Path for output CSV (default: {PROCESSED_DATA_DIR / 'final_ensemble_predictions.csv'})",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)  if args.input  else PROCESSED_DATA_DIR / "eval_dataset.csv"
    output_path = Path(args.output) if args.output else PROCESSED_DATA_DIR / "final_ensemble_predictions.csv"

    print("=" * 70)
    print("BUILD ENSEMBLE PREDICTIONS")
    print("=" * 70)
    print(f"Formula  : {W_FINANCIAL} * financial  +  {W_NEWS_TFIDF} * news_tfidf  +  {W_NEWS_EMBEDDINGS} * news_embeddings")
    print(f"Input    : {input_path}")
    print(f"Output   : {output_path}")
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Step 1/3  Loading eval_dataset.csv …")
    df = load_eval_dataset(input_path)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")
    print(f"  Date range: {df['prediction_date'].min()} → {df['prediction_date'].max()}")

    # ── Compute ───────────────────────────────────────────────────────────────
    print("Step 2/3  Computing ensemble …")
    out = compute_ensemble(df)

    up   = (out["ensemble_pred_binary"] == 1).sum()
    down = (out["ensemble_pred_binary"] == 0).sum()
    print(f"  Predictions: {up} up / {down} down")
    print(f"  Avg confidence: {out['ensemble_confidence'].mean():.3f}")
    if "actual_binary" in out.columns and out["actual_binary"].notna().any():
        acc = (out["ensemble_pred_binary"] == out["actual_binary"]).mean()
        print(f"  Accuracy (all rows): {acc:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("Step 3/3  Saving …")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  Saved {len(out):,} rows to {output_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Rows             : {len(out):,}")
    print(f"  Tickers          : {out['ticker'].nunique()}")
    if "split" in out.columns:
        for split, grp in out.groupby("split"):
            print(f"  {split.upper():5s}          : {len(grp):,} rows")
    print(f"  model_version    : {MODEL_VERSION}")
    print(f"  output           : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()