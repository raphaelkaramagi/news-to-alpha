#!/usr/bin/env python3
"""
TF-IDF NLP baseline: train on cutoff-aligned ticker-day rows and export predictions.

Outputs
-------
  data/models/nlp_baseline.joblib               – trained pipeline (vectorizer + LR)
  data/processed/news_tfidf_predictions.csv     – per ticker-day predictions

Cutoff contract
---------------
  News published before 4 PM ET on day D is assigned to prediction_date = D+1
  (the next trading day).  Headlines published on or after 4 PM ET on day D are
  assigned to prediction_date = D+1 (same rule — they didn't clear before close).
  Only labels with a matching prediction_date are used.

Usage
-----
  python scripts/train_nlp.py
  python scripts/train_nlp.py --db data/database.db
  python scripts/train_nlp.py --no-db-export   # skip writing to predictions table
"""

import sys
import json
import logging
import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
# Script lives at scripts/train_nlp.py  →  project root is one level up.
# We resolve relative to THIS file so it works regardless of cwd.
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Allow the script to also be called from project root directly
if not (_PROJECT_ROOT / "src").exists():
    _PROJECT_ROOT = _SCRIPT_DIR          # fallback: script IS in project root

sys.path.insert(0, str(_PROJECT_ROOT))

# Try to import config for canonical paths; fall back to sensible defaults.
try:
    from src.config import (  # noqa: E402
        DATABASE_PATH,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        TICKERS,
        CUTOFF_TIMEZONE,
        CUTOFF_HOUR,
    )
except ModuleNotFoundError:
    DATABASE_PATH     = _PROJECT_ROOT / "data" / "database.db"
    PROCESSED_DATA_DIR = _PROJECT_ROOT / "data" / "processed"
    MODELS_DIR        = _PROJECT_ROOT / "data" / "models"
    TICKERS: list[str] = [
        "AAPL", "NVDA", "WMT",  "LLY",  "JPM",
        "XOM",  "MCD",  "TSLA", "DAL",  "MAR",
        "GS",   "NFLX", "META", "ORCL", "PLTR",
    ]
    CUTOFF_TIMEZONE = "US/Eastern"
    CUTOFF_HOUR     = 16

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME    = "news_tfidf"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
MODEL_FILE    = "nlp_baseline.joblib"          # canonical, overwritten each run

PLACEHOLDER_TEXT = "__NO_NEWS__"               # sentinel for ticker-days with no headlines

ET = pytz.timezone(CUTOFF_TIMEZONE)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_labels(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load labels table.  Returns DataFrame with ticker, prediction_date, label_binary."""
    df = pd.read_sql_query(
        "SELECT ticker, date AS prediction_date, label_binary FROM labels",
        conn,
    )
    if df.empty:
        return df
    df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.date.astype(str)
    return df


def _published_at_to_prediction_date(published_at_str: str) -> Optional[str]:
    """
    Convert a published_at ISO-8601 string to a prediction_date string.

    Cutoff rule
    -----------
    News published BEFORE 4 PM ET on trading day D → prediction_date = next trading day.
    News published AT or AFTER 4 PM ET on day D → prediction_date = next trading day.

    In practice: we shift every article to the *next calendar day* relative to
    whichever ET date it falls on.  The labels table already encodes the trading-day
    sequence (consecutive rows in prices), so joining on prediction_date handles
    weekends / holidays automatically — articles that fall on a Friday 3 PM ET land
    on Saturday as a "prediction_date", find no label, and are dropped.  Articles
    that fall on Friday after 4 PM ET land on Saturday too, same result.  Only
    articles whose next-day lands on a trading day (i.e., a date present in labels)
    survive the join.

    Returns None if the timestamp cannot be parsed.
    """
    if not published_at_str:
        return None
    try:
        # Try ISO-8601 with offset first, then naive UTC assumption
        try:
            dt_utc = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
        except ValueError:
            # Finnhub sometimes returns UNIX timestamps stored as strings
            ts = float(published_at_str)
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)

        # Convert to Eastern
        dt_et = dt_utc.astimezone(ET)

        # Advance by one calendar day to get the prediction date
        pred_date = (dt_et + timedelta(days=1)).date()
        return str(pred_date)
    except Exception:
        return None


def _load_news_aligned(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load news, apply the 4 PM ET cutoff shift, and return one row per
    (ticker, prediction_date) with aggregated headlines.

    Returns an empty DataFrame (with correct columns) if there is no news.
    """
    news_raw = pd.read_sql_query(
        "SELECT ticker, title, published_at FROM news WHERE title IS NOT NULL AND title != ''",
        conn,
    )

    empty = pd.DataFrame(columns=["ticker", "prediction_date", "headlines_text", "top_headlines"])

    if news_raw.empty:
        log.warning("news table is empty — model will train on placeholder features only.")
        return empty

    # Apply cutoff shift
    news_raw["prediction_date"] = news_raw["published_at"].apply(_published_at_to_prediction_date)
    news_raw = news_raw.dropna(subset=["prediction_date"])

    if news_raw.empty:
        log.warning("All news rows had unparseable timestamps — no news features available.")
        return empty

    # Aggregate per (ticker, prediction_date)
    def _agg(group: pd.DataFrame) -> pd.Series:
        titles = group["title"].dropna().tolist()
        return pd.Series({
            "headlines_text": " . ".join(titles),   # single string fed to TF-IDF
            "top_headlines":  json.dumps(titles),   # JSON list stored in CSV
        })

    aggregated = (
        news_raw
        .groupby(["ticker", "prediction_date"], as_index=False)
        .apply(_agg, include_groups=False)
        .reset_index(drop=True)
    )
    # groupby(...).apply may produce a MultiIndex — flatten
    if isinstance(aggregated.index, pd.MultiIndex):
        aggregated = aggregated.reset_index()

    return aggregated


def build_dataset(db_path: str | Path) -> pd.DataFrame:
    """
    Join labels with cutoff-aligned news to produce the modeling dataset.

    Ticker-day rows without any news receive PLACEHOLDER_TEXT so the model
    can still be fit/predicted on every labeled row.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        labels = _load_labels(conn)
        news   = _load_news_aligned(conn)
    finally:
        conn.close()

    if labels.empty:
        raise RuntimeError(
            "labels table is empty.  Run scripts/generate_labels.py first."
        )

    # Left-join so every labeled ticker-day is represented
    if news.empty:
        df = labels.copy()
        df["headlines_text"] = PLACEHOLDER_TEXT
        df["top_headlines"]  = json.dumps([])
    else:
        df = labels.merge(news, on=["ticker", "prediction_date"], how="left")
        df["headlines_text"] = df["headlines_text"].fillna(PLACEHOLDER_TEXT)
        df["top_headlines"]  = df["top_headlines"].fillna(json.dumps([]))

    df = df.dropna(subset=["label_binary"])
    df["label_binary"] = df["label_binary"].astype(int)

    log.info(
        "Dataset: %d rows, %d tickers, %d with real news (%.0f%%)",
        len(df),
        df["ticker"].nunique(),
        (df["headlines_text"] != PLACEHOLDER_TEXT).sum(),
        100 * (df["headlines_text"] != PLACEHOLDER_TEXT).mean(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPLIT (chronological, matching the project contract)
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique prediction_date in chronological order.
    All tickers on the same date stay in the same split (no leakage).
    """
    dates = sorted(df["prediction_date"].unique())
    n     = len(dates)

    if n < 3:
        raise ValueError(
            f"Need at least 3 unique prediction_dates to split; got {n}. "
            "Run scripts/generate_labels.py and scripts/collect_prices.py to add more data."
        )

    train_end = int(n * train_ratio)
    val_end   = train_end + max(1, int(n * val_ratio))
    val_end   = min(val_end, n - 1)          # guarantee at least one test date

    train_dates = set(dates[:train_end])
    val_dates   = set(dates[train_end:val_end])
    test_dates  = set(dates[val_end:])

    train = df[df["prediction_date"].isin(train_dates)].copy()
    val   = df[df["prediction_date"].isin(val_dates)].copy()
    test  = df[df["prediction_date"].isin(test_dates)].copy()

    log.info(
        "Split sizes — train: %d rows (%d dates), val: %d rows (%d dates), test: %d rows (%d dates)",
        len(train), len(train_dates),
        len(val),   len(val_dates),
        len(test),  len(test_dates),
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(max_features: int = 5_000) -> Pipeline:
    """TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features    = max_features,
            ngram_range     = (1, 2),
            sublinear_tf    = True,
            strip_accents   = "unicode",
            min_df          = 2,           # ignore terms that appear in <2 docs
        )),
        ("clf", LogisticRegression(
            C             = 1.0,
            max_iter      = 1_000,
            class_weight  = "balanced",
            random_state  = 42,
            solver        = "lbfgs",
        )),
    ])


def train(
    train_df: pd.DataFrame,
    max_features: int = 5_000,
) -> Pipeline:
    """Fit the pipeline on training rows."""
    pipe = build_pipeline(max_features)
    pipe.fit(train_df["headlines_text"], train_df["label_binary"])
    log.info("Model trained on %d examples.", len(train_df))
    return pipe


def evaluate(pipe: Pipeline, df: pd.DataFrame, split_name: str) -> dict:
    """Return accuracy and ROC-AUC for a split; log results."""
    if df.empty:
        return {}
    proba  = pipe.predict_proba(df["headlines_text"])[:, 1]
    preds  = (proba >= 0.5).astype(int)
    acc    = accuracy_score(df["label_binary"], preds)
    try:
        auc = roc_auc_score(df["label_binary"], proba)
    except ValueError:
        auc = float("nan")
    log.info("%s — accuracy: %.3f  AUC: %.3f  (n=%d)", split_name, acc, auc, len(df))
    return {"accuracy": acc, "auc": auc, "n": len(df)}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_predictions(
    pipe:            Pipeline,
    split_name:      str,
    texts_split:     pd.Series,
    y_split:         pd.Series,
    metadata_split:  pd.DataFrame,
    raw_headlines_split: pd.Series,
) -> pd.DataFrame:
    """
    Produce a prediction DataFrame for one split.

    Parameters
    ----------
    pipe                : fitted sklearn Pipeline
    split_name          : "train" | "val" | "test"
    texts_split         : Series of headline strings fed to TF-IDF
    y_split             : Series of ground-truth labels (int 0/1)
    metadata_split      : DataFrame with at minimum columns [ticker, prediction_date]
    raw_headlines_split : Series of JSON-list strings (top_headlines column)

    Returns
    -------
    DataFrame with the required output columns.
    """
    proba  = pipe.predict_proba(texts_split)[:, 1]
    binary = (proba >= 0.5).astype(int)
    conf   = np.abs(proba - 0.5) * 2          # 0 = maximally uncertain, 1 = certain

    out = metadata_split[["ticker", "prediction_date"]].copy().reset_index(drop=True)
    out["split"]            = split_name
    out["model_name"]       = MODEL_NAME
    out["news_pred_proba"]  = proba
    out["news_pred_binary"] = binary
    out["news_confidence"]  = conf
    out["top_headlines"]    = raw_headlines_split.reset_index(drop=True)
    out["actual_binary"]    = y_split.reset_index(drop=True)
    out["model_version"]    = MODEL_VERSION

    return out


def save_predictions_csv(
    pipe:     Pipeline,
    splits:   dict[str, pd.DataFrame],
    out_path: Path,
) -> Path:
    """
    Run export_predictions for every split, concatenate, and save to CSV.

    Parameters
    ----------
    splits : {"train": df_train, "val": df_val, "test": df_test}
    """
    frames = []
    for split_name, df in splits.items():
        if df.empty:
            continue
        chunk = export_predictions(
            pipe             = pipe,
            split_name       = split_name,
            texts_split      = df["headlines_text"],
            y_split          = df["label_binary"],
            metadata_split   = df,
            raw_headlines_split = df["top_headlines"],
        )
        frames.append(chunk)

    if not frames:
        raise RuntimeError("No prediction rows generated — all splits are empty.")

    combined = pd.concat(frames, ignore_index=True)

    # Enforce column order
    col_order = [
        "ticker", "prediction_date", "split", "model_name",
        "news_pred_proba", "news_pred_binary", "news_confidence",
        "top_headlines", "actual_binary", "model_version",
    ]
    combined = combined[col_order]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    log.info("Predictions saved to %s  (%d rows)", out_path, len(combined))
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OPTIONAL: upsert into the predictions DB table
# ─────────────────────────────────────────────────────────────────────────────

def upsert_predictions_db(
    pipe:     Pipeline,
    splits:   dict[str, pd.DataFrame],
    db_path:  str | Path,
) -> int:
    """
    Insert (or replace) rows into the predictions table.
    Only fills news_* columns; financial_* and ensemble_* remain NULL.

    Returns number of rows upserted.
    """
    frames = []
    for split_name, df in splits.items():
        if df.empty:
            continue
        chunk = export_predictions(
            pipe                = pipe,
            split_name          = split_name,
            texts_split         = df["headlines_text"],
            y_split             = df["label_binary"],
            metadata_split      = df,
            raw_headlines_split = df["top_headlines"],
        )
        frames.append(chunk)

    if not frames:
        return 0

    combined = pd.concat(frames, ignore_index=True)

    conn  = sqlite3.connect(str(db_path))
    count = 0
    try:
        for _, row in combined.iterrows():
            conn.execute(
                """INSERT INTO predictions
                       (ticker, date, news_pred_proba, news_confidence,
                        news_top_headlines, news_pred_binary,
                        actual_binary, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, date, model_version) DO UPDATE SET
                       news_pred_proba    = excluded.news_pred_proba,
                       news_confidence    = excluded.news_confidence,
                       news_top_headlines = excluded.news_top_headlines,
                       news_pred_binary   = excluded.news_pred_binary,
                       actual_binary      = excluded.actual_binary
                """,
                (
                    row["ticker"],
                    row["prediction_date"],
                    float(row["news_pred_proba"]),
                    float(row["news_confidence"]),
                    row["top_headlines"],
                    int(row["news_pred_binary"]),
                    int(row["actual_binary"]),
                    row["model_version"],
                ),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()

    log.info("Upserted %d rows into predictions table.", count)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF NLP baseline")
    parser.add_argument(
        "--db", default=str(DATABASE_PATH),
        help=f"Path to SQLite database (default: {DATABASE_PATH})",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Fraction of dates for training (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Fraction of dates for validation (default: 0.15)",
    )
    parser.add_argument(
        "--max-features", type=int, default=5_000,
        help="TF-IDF vocabulary size (default: 5000)",
    )
    parser.add_argument(
        "--no-db-export", action="store_true",
        help="Skip writing predictions to the database predictions table",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override CSV output path",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    db_path  = Path(args.db)
    csv_path = Path(args.output) if args.output else (
        PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"
    )
    model_path = MODELS_DIR / MODEL_FILE
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TF-IDF NLP BASELINE")
    print("=" * 70)
    print(f"DB            : {db_path}")
    print(f"Model output  : {model_path}")
    print(f"CSV output    : {csv_path}")
    print(f"Split ratios  : {args.train_ratio:.0%} / {args.val_ratio:.0%} / "
          f"{1 - args.train_ratio - args.val_ratio:.0%}")
    print()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("Step 1/4  Loading data …")
    df = build_dataset(db_path)
    print(f"  {len(df):,} ticker-day rows, "
          f"{(df['headlines_text'] != PLACEHOLDER_TEXT).sum():,} with real news")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    print("Step 2/4  Splitting (chronological) …")
    train_df, val_df, test_df = chronological_split(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}

    # ── 3. Train ─────────────────────────────────────────────────────────────
    print("Step 3/4  Training …")
    pipe = train(train_df, max_features=args.max_features)

    # ── 4a. Evaluate ─────────────────────────────────────────────────────────
    print()
    metrics = {}
    for name, split_df in splits.items():
        metrics[name] = evaluate(pipe, split_df, name)

    # ── 4b. Save model ───────────────────────────────────────────────────────
    print(f"\nStep 4/4  Saving outputs …")
    joblib.dump(pipe, model_path)
    print(f"  Model saved  : {model_path}")

    # ── 4c. Export CSV ───────────────────────────────────────────────────────
    save_predictions_csv(pipe, splits, csv_path)
    print(f"  CSV saved    : {csv_path}")

    # ── 4d. Upsert DB (optional) ─────────────────────────────────────────────
    if not args.no_db_export:
        n = upsert_predictions_db(pipe, splits, db_path)
        print(f"  DB rows      : {n} upserted into predictions table")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, m in metrics.items():
        if m:
            print(f"  {name.upper():5s}  accuracy={m['accuracy']:.3f}  "
                  f"AUC={m['auc']:.3f}  n={m['n']}")
    print()
    print(f"  model_name    : {MODEL_NAME}")
    print(f"  model_version : {MODEL_VERSION}")
    print(f"  model_file    : {model_path}")
    print(f"  predictions   : {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()