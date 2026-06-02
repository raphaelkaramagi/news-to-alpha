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

import numpy as np
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
    "expected_move_pct",
    "actual_abs_return_pct",
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

    vol_path = PROCESSED_DATA_DIR / "volatility_predictions.csv"
    vol = pd.DataFrame(columns=["ticker", "prediction_date", "expected_move_pct", "actual_abs_return_pct"])
    if vol_path.exists():
        vol = pd.read_csv(vol_path)[
            ["ticker", "prediction_date", "expected_move_pct", "actual_abs_return_pct"]
        ]
        print(f"Loading volatility predictions ...")
        print(f"  {len(vol):,} rows")

    print("\nLeft-joining on (ticker, prediction_date) anchored on LSTM ...")
    df = price.merge(tfidf, on=["ticker", "prediction_date"], how="left")
    df = df.merge(embeddings, on=["ticker", "prediction_date"], how="left")
    if not vol.empty:
        df = df.merge(vol, on=["ticker", "prediction_date"], how="left")

    # has_news is derived from DB headline count (authoritative).
    import sqlite3
    from src.models.news_pipeline import _load_news_aligned  # noqa: E402
    from src.config import DATABASE_PATH  # noqa: E402

    head_df = pd.DataFrame(columns=["ticker", "prediction_date", "n_headlines"])
    if DATABASE_PATH.exists():
        conn = sqlite3.connect(str(DATABASE_PATH))
        try:
            news = _load_news_aligned(conn)
        finally:
            conn.close()
        if not news.empty:
            head_df = (
                news.rename(columns={"label_date": "prediction_date"})
                [["ticker", "prediction_date", "n_headlines"]]
            )
    if not head_df.empty:
        df = df.merge(head_df, on=["ticker", "prediction_date"], how="left")
    df["n_headlines"] = df.get("n_headlines", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["has_news"] = (df["n_headlines"] > 0).astype(int)

    # Fill missing news probas: neutral only when there is no news
    for col, neutral in [
        ("news_tfidf_pred_proba", 0.5),
        ("news_embeddings_pred_proba", 0.5),
        ("news_tfidf_confidence", 0.0),
        ("news_embeddings_confidence", 0.0),
    ]:
        if col in df.columns:
            no_news = df["has_news"] == 0
            df.loc[no_news, col] = df.loc[no_news, col].fillna(neutral)
            if col == "news_embeddings_pred_proba" and "news_tfidf_pred_proba" in df.columns:
                miss = (df["has_news"] == 1) & df[col].isna()
                df.loc[miss, col] = df.loc[miss, "news_tfidf_pred_proba"]
            if col == "news_tfidf_pred_proba" and "news_embeddings_pred_proba" in df.columns:
                miss = (df["has_news"] == 1) & df[col].isna()
                df.loc[miss, col] = df.loc[miss, "news_embeddings_pred_proba"]
            still = (df["has_news"] == 1) & df[col].isna()
            if still.any():
                df.loc[still, col] = neutral

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

    # Only use news scores from the matching chronological split to avoid
    # applying train-set news predictions onto LSTM test rows (data leakage).
    split_match = (
        (df["price_split"] == df["news_tfidf_split"])
        | df["news_tfidf_split"].isna()
    )
    for col in (
        "news_tfidf_pred_proba",
        "news_tfidf_pred_binary",
        "news_tfidf_confidence",
        "news_tfidf_top_headlines",
    ):
        if col in df.columns:
            df.loc[~split_match, col] = np.nan

    emb_match = (
        (df["price_split"] == df["news_embeddings_split"])
        | df["news_embeddings_split"].isna()
    )
    for col in (
        "news_embeddings_pred_proba",
        "news_embeddings_pred_binary",
        "news_embeddings_confidence",
        "news_embeddings_top_headlines",
    ):
        if col in df.columns:
            df.loc[~emb_match, col] = np.nan

    print(f"  Rows after join: {len(df):,}")
    print(f"  Tickers        : {df['ticker'].nunique()}")
    print(f"  Date range     : {df['prediction_date'].min()} -> {df['prediction_date'].max()}")
    print(f"  Rows with news : {int(df['has_news'].sum()):,}  "
          f"({df['has_news'].mean():.0%})")

    out_cols = [c for c in OUTPUT_COLS if c in df.columns]
    df = df[out_cols]
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "prediction_date"], keep="last")
    if len(df) < before:
        print(f"  Deduped {before - len(df)} duplicate (ticker, date) rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} rows to {output_path}")
    print("\nSplit breakdown:")
    for split, grp in df.groupby("split"):
        print(f"  {split.upper():5s}: {len(grp):,} rows")
    print("=" * 70)


if __name__ == "__main__":
    main()
