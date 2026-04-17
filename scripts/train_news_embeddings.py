#!/usr/bin/env python3
"""
Advanced news model: sentence embeddings + logistic regression on cutoff-aligned data.

Uses the same ticker-day text preparation as scripts/train_nlp.py (build_dataset):
labels joined on (ticker, prediction_date) with headlines aggregated after the
4 PM ET cutoff mapping — not raw published_at calendar dates alone.

Outputs
-------
  data/models/news_embeddings.joblib          – classifier + sentence model name
  data/processed/news_embeddings_predictions.csv

Usage
-----
  python scripts/train_news_embeddings.py
  python scripts/train_news_embeddings.py --sentence-model all-MiniLM-L6-v2
  python scripts/train_news_embeddings.py --no-db-export

First run downloads the sentence model from Hugging Face. If ``HF_HOME`` is not set,
the script uses ``data/.hf_cache`` under the project so the cache stays writable
(e.g. in sandboxes or read-only home directories).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
# ── Project paths ───────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if not (_PROJECT_ROOT / "src").exists():
    _PROJECT_ROOT = _SCRIPT_DIR

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import train_nlp as tn  # noqa: E402  — same dataset contract as TF-IDF baseline

try:
    from src.config import DATABASE_PATH, MODELS_DIR, PROCESSED_DATA_DIR  # noqa: E402
except ModuleNotFoundError:
    DATABASE_PATH = _PROJECT_ROOT / "data" / "database.db"
    MODELS_DIR = _PROJECT_ROOT / "data" / "models"
    PROCESSED_DATA_DIR = _PROJECT_ROOT / "data" / "processed"

log = logging.getLogger(__name__)

MODEL_NAME = "news_embeddings"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
MODEL_FILE = "news_embeddings.joblib"
DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"
CSV_NAME = "news_embeddings_predictions.csv"


def _ensure_hf_home_for_downloads() -> None:
    """If HF_HOME is unset, use data/.hf_cache under the project (writable in CI/sandbox)."""
    if os.environ.get("HF_HOME"):
        return
    cache = _PROJECT_ROOT / "data" / ".hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)


class NewsEmbeddingClassifier:
    """
    Sentence-encoder (by Hugging Face name) + sklearn logistic regression.
    Encoder is loaded lazily so joblib files stay small (only clf + name).
    """

    def __init__(
        self,
        sentence_model_name: str,
        classifier: LogisticRegression,
        batch_size: int = 32,
    ) -> None:
        self.sentence_model_name = sentence_model_name
        self.classifier = classifier
        self.batch_size = batch_size
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.sentence_model_name)
        return self._encoder

    def encode(
        self,
        texts: pd.Series | list[str],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, pd.Series):
            text_list = texts.astype(str).tolist()
        else:
            text_list = [str(t) for t in texts]
        enc = self._get_encoder()
        return enc.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def fit(self, train_texts: pd.Series, y: np.ndarray | pd.Series) -> NewsEmbeddingClassifier:
        log.info("Encoding %d training texts …", len(train_texts))
        X_train = self.encode(train_texts, show_progress=True)
        y_arr = np.asarray(y).astype(int)
        self.classifier.fit(X_train, y_arr)
        log.info("Classifier fitted on embedding dim %d.", X_train.shape[1])
        return self

    def predict_proba_positive(self, texts: pd.Series, *, show_progress: bool = False) -> np.ndarray:
        X = self.encode(texts, show_progress=show_progress)
        return self.classifier.predict_proba(X)[:, 1]


def evaluate(model: NewsEmbeddingClassifier, df: pd.DataFrame, split_name: str) -> dict:
    if df.empty:
        return {}
    proba = model.predict_proba_positive(df["headlines_text"], show_progress=len(df) > 200)
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(df["label_binary"], preds)
    try:
        auc = roc_auc_score(df["label_binary"], proba)
    except ValueError:
        auc = float("nan")
    log.info("%s — accuracy: %.3f  AUC: %.3f  (n=%d)", split_name, acc, auc, len(df))
    return {"accuracy": acc, "auc": auc, "n": len(df)}


def export_predictions(
    model: NewsEmbeddingClassifier,
    split_name: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    proba = model.predict_proba_positive(df["headlines_text"], show_progress=len(df) > 200)
    binary = (proba >= 0.5).astype(int)
    conf = np.abs(proba - 0.5) * 2

    out = df[["ticker", "prediction_date"]].copy().reset_index(drop=True)
    out["split"] = split_name
    out["model_name"] = MODEL_NAME
    out["news_pred_proba"] = proba
    out["news_pred_binary"] = binary
    out["news_confidence"] = conf
    out["top_headlines"] = df["top_headlines"].reset_index(drop=True)
    out["actual_binary"] = df["label_binary"].reset_index(drop=True)
    out["model_version"] = MODEL_VERSION
    return out


def save_predictions_csv(
    model: NewsEmbeddingClassifier,
    splits: dict[str, pd.DataFrame],
    out_path: Path,
) -> Path:
    frames = []
    for split_name, split_df in splits.items():
        if split_df.empty:
            continue
        frames.append(export_predictions(model, split_name, split_df))
    if not frames:
        raise RuntimeError("No prediction rows generated — all splits are empty.")
    combined = pd.concat(frames, ignore_index=True)
    col_order = [
        "ticker",
        "prediction_date",
        "split",
        "model_name",
        "news_pred_proba",
        "news_pred_binary",
        "news_confidence",
        "top_headlines",
        "actual_binary",
        "model_version",
    ]
    combined = combined[col_order]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    log.info("Predictions saved to %s  (%d rows)", out_path, len(combined))
    return out_path


def upsert_predictions_db(
    model: NewsEmbeddingClassifier,
    splits: dict[str, pd.DataFrame],
    db_path: str | Path,
) -> int:
    frames = []
    for split_name, split_df in splits.items():
        if split_df.empty:
            continue
        frames.append(export_predictions(model, split_name, split_df))
    if not frames:
        return 0
    combined = pd.concat(frames, ignore_index=True)
    conn = sqlite3.connect(str(db_path))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train sentence-embedding news model (Session 1 cutoff-aligned rows)",
    )
    parser.add_argument(
        "--db",
        default=str(DATABASE_PATH),
        help=f"SQLite database path (default: {DATABASE_PATH})",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--sentence-model",
        default=DEFAULT_SENTENCE_MODEL,
        help=f"Hugging Face model id for sentence-transformers (default: {DEFAULT_SENTENCE_MODEL})",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--no-db-export", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Override CSV output path",
    )
    args = parser.parse_args()

    _ensure_hf_home_for_downloads()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    db_path = Path(args.db)
    csv_path = Path(args.output) if args.output else (PROCESSED_DATA_DIR / CSV_NAME)
    model_path = MODELS_DIR / MODEL_FILE
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEWS EMBEDDINGS MODEL (cutoff-aligned ticker-day rows)")
    print("=" * 70)
    print(f"DB               : {db_path}")
    print(f"Sentence model   : {args.sentence_model}")
    print(f"Model output     : {model_path}")
    print(f"CSV output       : {csv_path}")
    print(f"Dataset builder  : train_nlp.build_dataset (prediction_date key)")
    print()

    print("Step 1/4  Loading data …")
    df_all = tn.build_dataset(db_path)
    df = df_all[df_all["headlines_text"] != tn.PLACEHOLDER_TEXT].copy()

    print(f"  {len(df_all):,} ticker-day rows total")
    print(f"  {len(df):,} with real news")

    if df.empty:
        raise RuntimeError("No real-news rows available for training.")

    print("Step 2/4  Splitting (chronological) …")
    train_df, val_df, test_df = tn.chronological_split(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}

    clf = LogisticRegression(
        C=1.0,
        max_iter=1_000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    model = NewsEmbeddingClassifier(
        sentence_model_name=args.sentence_model,
        classifier=clf,
        batch_size=args.batch_size,
    )

    print("Step 3/4  Training (encode train + fit classifier) …")
    model.fit(train_df["headlines_text"], train_df["label_binary"])

    print()
    metrics: dict[str, dict] = {}
    for name, split_df in splits.items():
        metrics[name] = evaluate(model, split_df, name)

    print("\nStep 4/4  Saving outputs …")
    payload = {
        "classifier": model.classifier,
        "sentence_model_name": model.sentence_model_name,
        "batch_size": model.batch_size,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset": "train_nlp.build_dataset",
    }
    joblib.dump(payload, model_path)
    print(f"  Model saved  : {model_path}")

    save_predictions_csv(model, splits, csv_path)
    print(f"  CSV saved    : {csv_path}")

    meta_path = model_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "sentence_model": args.sentence_model,
                "csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Meta saved   : {meta_path}")

    if not args.no_db_export:
        n = upsert_predictions_db(model, splits, db_path)
        print(f"  DB rows      : {n} upserted into predictions table")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, m in metrics.items():
        if m:
            print(
                f"  {name.upper():5s}  accuracy={m['accuracy']:.3f}  "
                f"AUC={m['auc']:.3f}  n={m['n']}",
            )
    print()
    print(f"  model_name    : {MODEL_NAME}")
    print(f"  model_version : {MODEL_VERSION}")
    print("=" * 70)


if __name__ == "__main__":
    main()
