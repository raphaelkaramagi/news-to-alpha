#!/usr/bin/env python3
"""
Build the aligned evaluation dataset by joining all three model prediction CSVs.

Inputs
------
  data/processed/price_predictions.csv
  data/processed/news_tfidf_predictions.csv
  data/processed/news_embeddings_predictions.csv

Output
------
  data/processed/eval_dataset.csv

Join key : (ticker, prediction_date)
Split    : price_split is used as the canonical split column (per session_2_contract.md)

Usage
-----
  python scripts/build_eval_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

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

# ── Required output columns ───────────────────────────────────────────────────
OUTPUT_COLS = [
    "ticker",
    "prediction_date",
    "split",
    "price_split",
    "news_tfidf_split",
    "news_embeddings_split",
    "financial_pred_proba",
    "financial_pred_binary",
    "financial_confidence",
    "news_tfidf_pred_proba",
    "news_tfidf_pred_binary",
    "news_tfidf_confidence",
    "news_embeddings_pred_proba",
    "news_embeddings_pred_binary",
    "news_embeddings_confidence",
    "top_headlines",
    "actual_binary",
]


def load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"split": "price_split"})
    return df[[
        "ticker", "prediction_date", "price_split",
        "financial_pred_proba", "financial_pred_binary", "financial_confidence",
        "actual_binary",
    ]]


def load_news_tfidf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "split":            "news_tfidf_split",
        "news_pred_proba":  "news_tfidf_pred_proba",
        "news_pred_binary": "news_tfidf_pred_binary",
        "news_confidence":  "news_tfidf_confidence",
        "top_headlines":    "news_tfidf_top_headlines",
    })
    return df[[
        "ticker", "prediction_date", "news_tfidf_split",
        "news_tfidf_pred_proba", "news_tfidf_pred_binary", "news_tfidf_confidence",
        "news_tfidf_top_headlines",
    ]]


def load_news_embeddings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "split":            "news_embeddings_split",
        "news_pred_proba":  "news_embeddings_pred_proba",
        "news_pred_binary": "news_embeddings_pred_binary",
        "news_confidence":  "news_embeddings_confidence",
        "top_headlines":    "news_embeddings_top_headlines",
    })
    return df[[
        "ticker", "prediction_date", "news_embeddings_split",
        "news_embeddings_pred_proba", "news_embeddings_pred_binary", "news_embeddings_confidence",
        "news_embeddings_top_headlines",
    ]]


def main() -> None:
    price_path      = PROCESSED_DATA_DIR / "price_predictions.csv"
    tfidf_path      = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"
    embeddings_path = PROCESSED_DATA_DIR / "news_embeddings_predictions.csv"
    output_path     = PROCESSED_DATA_DIR / "eval_dataset.csv"

    print("=" * 70)
    print("BUILD EVAL DATASET")
    print("=" * 70)

    for p in [price_path, tfidf_path, embeddings_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    print("Loading price predictions …")
    price = load_price(price_path)
    print(f"  {len(price):,} rows")

    print("Loading news TF-IDF predictions …")
    tfidf = load_news_tfidf(tfidf_path)
    print(f"  {len(tfidf):,} rows")

    print("Loading news embeddings predictions …")
    embeddings = load_news_embeddings(embeddings_path)
    print(f"  {len(embeddings):,} rows")

    print("\nJoining on (ticker, prediction_date) …")
    df = price.merge(tfidf,      on=["ticker", "prediction_date"], how="inner")
    df = df.merge(embeddings,    on=["ticker", "prediction_date"], how="inner")

    print(f"  Rows after inner join: {len(df):,}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Date range: {df['prediction_date'].min()} → {df['prediction_date'].max()}")

    # top_headlines: prefer embeddings, fall back to tfidf
    df["top_headlines"] = df["news_embeddings_top_headlines"].fillna(
        df["news_tfidf_top_headlines"]
    )

    # canonical split = price_split
    df["split"] = df["price_split"]

    out_cols = [c for c in OUTPUT_COLS if c in df.columns]
    df = df[out_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df):,} rows to {output_path}")
    print("\nSplit breakdown:")
    for split, grp in df.groupby("split"):
        print(f"  {split.upper():5s}: {len(grp):,} rows")
    print("=" * 70)


if __name__ == "__main__":
    main()