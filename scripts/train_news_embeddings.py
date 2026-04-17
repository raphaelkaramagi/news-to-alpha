#!/usr/bin/env python3
"""Advanced news model: per-headline sentence embeddings + logistic regression.

Pipeline per ticker-day row:
1. Load the raw headline list (one string per headline).
2. Encode each headline individually with sentence-transformers (MiniLM).
3. Mean-pool across headlines -> one 384-d vector per row (single-headline
   semantics is preserved, no concatenation artefacts).
4. Optionally concatenate 12 FinBERT sentiment aggregates + a 16-d
   publisher one-hot vector.
5. Calibrated logistic regression on the concatenated features.

Uses `src.models.news_pipeline.build_dataset` so train / eval rows are
cutoff-aligned and exactly the same as the TF-IDF baseline.

Outputs
-------
  data/models/news_embeddings.joblib
  data/processed/news_embeddings_predictions.csv
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False


def _make_prefit_calibrator(estimator, method: str = "sigmoid") -> CalibratedClassifierCV:
    """Compat wrapper: sklearn>=1.6 needs FrozenEstimator; older uses cv='prefit'."""
    if _HAS_FROZEN:
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    return CalibratedClassifierCV(estimator, method=method, cv="prefit")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, MODELS_DIR, PROCESSED_DATA_DIR  # noqa: E402
from src.features.publisher_features import PublisherOneHot  # noqa: E402
from src.models.news_pipeline import build_dataset, chronological_split  # noqa: E402

log = logging.getLogger(__name__)

MODEL_NAME = "news_embeddings"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
MODEL_FILE = "news_embeddings.joblib"
DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"
CSV_NAME = "news_embeddings_predictions.csv"


def _ensure_hf_home() -> None:
    if os.environ.get("HF_HOME"):
        return
    cache = _PROJECT_ROOT / "data" / ".hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)


def _parse_headlines_col(cell: Any) -> list[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, list):
        return [str(s).strip() for s in cell if s]
    try:
        return [str(s).strip() for s in json.loads(cell) if s]
    except (ValueError, TypeError):
        return [str(cell)]


class NewsEmbeddingClassifier:
    """MiniLM per-headline pooling + (optional) FinBERT + publisher one-hot."""

    def __init__(
        self,
        sentence_model_name: str = DEFAULT_SENTENCE_MODEL,
        classifier: LogisticRegression | None = None,
        batch_size: int = 32,
        use_finbert: bool = False,
        finbert_cache_db: str | Path | None = None,
        publisher_encoder: PublisherOneHot | None = None,
    ) -> None:
        self.sentence_model_name = sentence_model_name
        self.classifier: Any = classifier or LogisticRegression(
            C=1.0, max_iter=1_000,
            class_weight="balanced", random_state=42, solver="lbfgs",
        )
        self.batch_size = batch_size
        self.use_finbert = use_finbert
        self.finbert_cache_db = str(finbert_cache_db) if finbert_cache_db else None
        self.publisher_encoder = publisher_encoder or PublisherOneHot(top_n=15)

        self._encoder: Any = None
        self._finbert: Any = None
        self._calibrated: Any = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.sentence_model_name)
        return self._encoder

    def _get_finbert(self):
        if self._finbert is None:
            from src.features.news_sentiment import FinBertSentiment

            self._finbert = FinBertSentiment(
                cache_db=self.finbert_cache_db,
                batch_size=self.batch_size,
            )
        return self._finbert

    def _encode_rows(
        self,
        headlines_per_row: list[list[str]],
        weights_per_row: list[list[float]] | None = None,
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Per-headline encode + relevance-weighted pooling -> (n_rows, emb_dim).

        Falls back to uniform mean when weights are zero / absent.
        """
        encoder = self._get_encoder()
        emb_dim = encoder.get_sentence_embedding_dimension()

        flat: list[str] = []
        slices: list[tuple[int, int]] = []
        for row in headlines_per_row:
            start = len(flat)
            flat.extend(row)
            slices.append((start, len(flat)))

        if flat:
            all_emb = encoder.encode(
                flat,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            ).astype(np.float32)
        else:
            all_emb = np.zeros((0, emb_dim), dtype=np.float32)

        out = np.zeros((len(headlines_per_row), emb_dim), dtype=np.float32)
        for i, (s, e) in enumerate(slices):
            if e <= s:
                continue
            emb = all_emb[s:e]
            if weights_per_row is not None:
                w = np.asarray(weights_per_row[i], dtype=np.float32)
                if w.shape[0] != emb.shape[0] or (w.sum() <= 1e-8):
                    out[i] = emb.mean(axis=0)
                else:
                    w = w / w.sum()
                    out[i] = (emb * w[:, None]).sum(axis=0)
            else:
                out[i] = emb.mean(axis=0)
        return out

    def build_features(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Assemble the full feature matrix for one split."""
        headlines_per_row = [_parse_headlines_col(c) for c in df["top_headlines"]]
        sources_per_row = df["sources"]
        # Per-headline relevance weights: single scalar per row broadcast across headlines.
        avg_rel = (
            pd.to_numeric(df["avg_relevance"], errors="coerce").fillna(0.0)
            if "avg_relevance" in df.columns
            else pd.Series(np.zeros(len(df)))
        )
        weights_per_row = [
            [float(r)] * len(headlines) if len(headlines) else []
            for r, headlines in zip(avg_rel, headlines_per_row)
        ]

        embeddings = self._encode_rows(
            headlines_per_row, weights_per_row, show_progress=show_progress,
        )

        if fit:
            pub_vec = self.publisher_encoder.fit_transform(sources_per_row)
        else:
            pub_vec = self.publisher_encoder.transform(sources_per_row)

        # Side features: n_headlines, avg_finnhub_sentiment, avg_relevance
        side = np.column_stack([
            pd.to_numeric(df.get("n_headlines", 0), errors="coerce").fillna(0).to_numpy(),
            pd.to_numeric(df.get("avg_finnhub_sentiment", 0), errors="coerce").fillna(0).to_numpy(),
            avg_rel.to_numpy(),
        ]).astype(np.float32)

        parts = [embeddings, pub_vec, side]
        if self.use_finbert:
            fb = self._get_finbert().score_ticker_days(headlines_per_row)
            parts.append(fb)

        return np.hstack(parts).astype(np.float32)

    def fit(self, train_df: pd.DataFrame) -> "NewsEmbeddingClassifier":
        log.info("Encoding %d training rows ...", len(train_df))
        X = self.build_features(train_df, fit=True, show_progress=True)
        y = train_df["label_binary"].to_numpy(dtype=int)
        self.classifier.fit(X, y)
        log.info("Classifier fitted on feature dim %d.", X.shape[1])
        return self

    def calibrate(self, val_df: pd.DataFrame) -> "NewsEmbeddingClassifier":
        if val_df.empty or val_df["label_binary"].nunique() < 2:
            log.warning("Skipping calibration (val empty / single-class).")
            self._calibrated = None
            return self
        X_val = self.build_features(val_df, fit=False)
        y_val = val_df["label_binary"].to_numpy(dtype=int)
        cal = _make_prefit_calibrator(self.classifier, method="sigmoid")
        cal.fit(X_val, y_val)
        self._calibrated = cal
        log.info("Calibrated embeddings classifier on %d val rows.", len(val_df))
        return self

    def predict_proba_positive(
        self, df: pd.DataFrame, *, show_progress: bool = False,
    ) -> np.ndarray:
        X = self.build_features(df, fit=False, show_progress=show_progress)
        clf = self._calibrated if self._calibrated is not None else self.classifier
        return clf.predict_proba(X)[:, 1]


def evaluate(model: NewsEmbeddingClassifier, df: pd.DataFrame, split_name: str) -> dict:
    if df.empty:
        return {}
    proba = model.predict_proba_positive(df, show_progress=len(df) > 200)
    preds = (proba >= 0.5).astype(int)
    acc = accuracy_score(df["label_binary"], preds)
    try:
        auc = roc_auc_score(df["label_binary"], proba)
    except ValueError:
        auc = float("nan")
    log.info("%s - accuracy: %.3f  AUC: %.3f  (n=%d)", split_name, acc, auc, len(df))
    return {"accuracy": acc, "auc": auc, "n": len(df)}


def export_predictions(model: NewsEmbeddingClassifier,
                       split_name: str, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba_positive(df)
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


def save_predictions_csv(model: NewsEmbeddingClassifier,
                         splits: dict[str, pd.DataFrame],
                         out_path: Path) -> Path:
    frames = [export_predictions(model, name, df)
              for name, df in splits.items() if not df.empty]
    if not frames:
        raise RuntimeError("No prediction rows generated - all splits empty.")
    combined = pd.concat(frames, ignore_index=True)
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


def upsert_predictions_db(model: NewsEmbeddingClassifier,
                          splits: dict[str, pd.DataFrame],
                          db_path: str | Path) -> int:
    frames = [export_predictions(model, name, df)
              for name, df in splits.items() if not df.empty]
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
                    row["ticker"], row["prediction_date"],
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
    parser = argparse.ArgumentParser(description="Train embeddings news model")
    parser.add_argument("--db", default=str(DATABASE_PATH))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-finbert", action="store_true",
                        help="Concat FinBERT sentiment aggregates to the features")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3],
                        help="Prediction horizon in trading days (default: 1)")
    parser.add_argument("--min-move-pct", type=float, default=0.0,
                        help="Drop TRAIN rows where |return_horizon| < this (eval keeps all).")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--no-db-export", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    _ensure_hf_home()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    db_path = Path(args.db)
    csv_path = Path(args.output) if args.output else (PROCESSED_DATA_DIR / CSV_NAME)
    model_path = MODELS_DIR / MODEL_FILE
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEWS EMBEDDINGS MODEL (per-headline mean-pool + publisher 1-hot"
          + (", +FinBERT" if args.use_finbert else "") + ")")
    print("=" * 70)
    print(f"DB               : {db_path}")
    print(f"Sentence model   : {args.sentence_model}")
    print(f"Model output     : {model_path}")
    print(f"CSV output       : {csv_path}")
    print(f"FinBERT features : {'yes' if args.use_finbert else 'no'}")
    print()

    print("Step 1/4  Loading data ...")
    df = build_dataset(db_path, drop_rows_without_news=True, horizon=args.horizon)
    print(f"  {len(df):,} ticker-day rows (news-aligned, horizon={args.horizon})")

    print("Step 2/4  Splitting (chronological) ...")
    train_df, val_df, test_df = chronological_split(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    if args.min_move_pct > 0 and not train_df.empty:
        before = len(train_df)
        train_df = train_df[train_df["label_return"].abs() >= args.min_move_pct].copy()
        print(f"  Drop-flat filter (|ret|<{args.min_move_pct}%): "
              f"dropped {before - len(train_df)} train rows, kept {len(train_df)}")
    splits = {"train": train_df, "val": val_df, "test": test_df}

    finbert_cache = (PROCESSED_DATA_DIR / "finbert_cache.db") if args.use_finbert else None
    model = NewsEmbeddingClassifier(
        sentence_model_name=args.sentence_model,
        batch_size=args.batch_size,
        use_finbert=args.use_finbert,
        finbert_cache_db=finbert_cache,
    )

    print("Step 3/4  Training ...")
    model.fit(train_df)

    if not args.no_calibration:
        model.calibrate(val_df)

    print()
    metrics = {name: evaluate(model, split_df, name) for name, split_df in splits.items()}

    print("\nStep 4/4  Saving outputs ...")
    payload = {
        "classifier": model.classifier,
        "calibrated": model._calibrated,
        "sentence_model_name": model.sentence_model_name,
        "publisher_encoder": model.publisher_encoder,
        "use_finbert": model.use_finbert,
        "finbert_cache_db": model.finbert_cache_db,
        "batch_size": model.batch_size,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset": "src.models.news_pipeline.build_dataset",
    }
    joblib.dump(payload, model_path)
    print(f"  Model saved  : {model_path}")

    save_predictions_csv(model, splits, csv_path)
    print(f"  CSV saved    : {csv_path}")

    if not args.no_db_export:
        n = upsert_predictions_db(model, splits, db_path)
        print(f"  DB rows      : {n} upserted into predictions table")

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
    print("=" * 70)


if __name__ == "__main__":
    main()
