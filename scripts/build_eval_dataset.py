#!/usr/bin/env python3
"""Build the aligned evaluation dataset by joining all three prediction CSVs.

Inputs
------
  data/processed/price_predictions.csv          (LSTM per ticker-day)
  data/processed/news_tfidf_predictions.csv     (TF-IDF; only news-bearing days)
  data/processed/news_embeddings_predictions.csv (Embeddings; only news-bearing days)

Output
------
  data/processed/eval_dataset.csv

Join policy
-----------
LSTM is the anchor (LEFT join from `price_predictions.csv`) so that every
(ticker, prediction_date) the LSTM scored survives.  News predictions are
filled with 0.5 / 0.0 / empty when missing, and a `has_news` flag is added.

The ensemble script downstream understands `has_news` and reweights (via its
meta model) accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR  # noqa: E402

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
    "has_news",
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
    if not path.exists():
        return pd.DataFrame(columns=[
            "ticker", "prediction_date", "news_tfidf_split",
            "news_tfidf_pred_proba", "news_tfidf_pred_binary",
            "news_tfidf_confidence", "news_tfidf_top_headlines",
        ])
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
        "news_tfidf_pred_proba", "news_tfidf_pred_binary",
        "news_tfidf_confidence", "news_tfidf_top_headlines",
    ]]


def load_news_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "ticker", "prediction_date", "news_embeddings_split",
            "news_embeddings_pred_proba", "news_embeddings_pred_binary",
            "news_embeddings_confidence", "news_embeddings_top_headlines",
        ])
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
        "news_embeddings_pred_proba", "news_embeddings_pred_binary",
        "news_embeddings_confidence", "news_embeddings_top_headlines",
    ]]


def main() -> None:
    price_path = PROCESSED_DATA_DIR / "price_predictions.csv"
    tfidf_path = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"
    embeddings_path = PROCESSED_DATA_DIR / "news_embeddings_predictions.csv"
    output_path = PROCESSED_DATA_DIR / "eval_dataset.csv"

    print("=" * 70)
    print("BUILD EVAL DATASET (LEFT join from LSTM predictions)")
    print("=" * 70)

    if not price_path.exists():
        raise FileNotFoundError(
            f"Missing {price_path}. Run scripts/train_lstm.py first."
        )

    print("Loading price predictions ...")
    price = load_price(price_path)
    print(f"  {len(price):,} rows")

    print("Loading news TF-IDF predictions ...")
    tfidf = load_news_tfidf(tfidf_path)
    print(f"  {len(tfidf):,} rows")

    print("Loading news embeddings predictions ...")
    embeddings = load_news_embeddings(embeddings_path)
    print(f"  {len(embeddings):,} rows")

    print("\nLeft-joining on (ticker, prediction_date) anchored on LSTM ...")
    df = price.merge(tfidf, on=["ticker", "prediction_date"], how="left")
    df = df.merge(embeddings, on=["ticker", "prediction_date"], how="left")

    # has_news = at least one news model produced a probability
    tfidf_present = df.get("news_tfidf_pred_proba", pd.Series(dtype=float)).notna()
    emb_present = df.get("news_embeddings_pred_proba", pd.Series(dtype=float)).notna()
    df["has_news"] = (tfidf_present | emb_present).astype(int)

    # Fill missing news probas with neutral 0.5 and zero confidence
    for col, neutral in [
        ("news_tfidf_pred_proba", 0.5),
        ("news_embeddings_pred_proba", 0.5),
        ("news_tfidf_confidence", 0.0),
        ("news_embeddings_confidence", 0.0),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(neutral)

    for col in ("news_tfidf_pred_binary", "news_embeddings_pred_binary"):
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)

    # Pick top_headlines: prefer embeddings, fall back to tfidf, else empty list.
    emb_top = df.get("news_embeddings_top_headlines")
    tfidf_top = df.get("news_tfidf_top_headlines")
    if emb_top is not None and tfidf_top is not None:
        df["top_headlines"] = emb_top.fillna(tfidf_top).fillna("[]")
    elif emb_top is not None:
        df["top_headlines"] = emb_top.fillna("[]")
    elif tfidf_top is not None:
        df["top_headlines"] = tfidf_top.fillna("[]")
    else:
        df["top_headlines"] = "[]"

    df["split"] = df["price_split"]

    print(f"  Rows after join: {len(df):,}")
    print(f"  Tickers        : {df['ticker'].nunique()}")
    print(f"  Date range     : {df['prediction_date'].min()} -> {df['prediction_date'].max()}")
    print(f"  Rows with news : {int(df['has_news'].sum()):,}  "
          f"({df['has_news'].mean():.0%})")

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
