#!/usr/bin/env python3
"""TF-IDF NLP baseline: headlines + publisher features -> calibrated LogReg.

Feature pipeline per ticker-day row:
    TfidfVectorizer(headlines_text)       sparse (n, max_features)
    PublisherOneHot(sources)              dense  (n, top_n+1)
    -> scipy hstack -> classifier

Uses `src.models.news_pipeline.build_dataset` so the cutoff rule is the
single source of truth and rows without news are excluded.

Outputs
-------
  data/models/nlp_baseline.joblib
  data/processed/news_tfidf_predictions.csv

Usage
-----
  python scripts/train_nlp.py
  python scripts/train_nlp.py --max-features 8000
  python scripts/train_nlp.py --no-db-export
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR, MODELS_DIR, NLP_CONFIG  # noqa: E402
from src.features.publisher_features import PublisherOneHot  # noqa: E402
from src.ml.threshold_tuning import (  # noqa: E402
    calibration_preserves_spread,
    tune_threshold_balanced_accuracy,
)
from src.models.news_pipeline import build_dataset, chronological_split, assign_split_labels  # noqa: E402
from src.config import PROCESSED_DATA_DIR  # noqa: E402

MODEL_NAME = "news_tfidf"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
MODEL_FILE = "nlp_baseline.joblib"
CSV_NAME = "news_tfidf_predictions.csv"

log = logging.getLogger(__name__)


_SIDE_FEATURES = ["n_headlines", "avg_finnhub_sentiment", "avg_relevance"]


def _text_with_snippet(df: pd.DataFrame) -> list[str]:
    """Concat headline strings + content snippet with a clear separator."""
    titles = df["headlines_text"].fillna("").astype(str)
    snippets = (
        df["content_snippet"].fillna("").astype(str)
        if "content_snippet" in df.columns else pd.Series("" * len(df), index=df.index)
    )
    return (titles + " ## " + snippets).tolist()


def _side_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for c in _SIDE_FEATURES:
        if c in df.columns:
            cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy())
        else:
            cols.append(np.zeros(len(df), dtype=float))
    return np.asarray(cols, dtype=float).T


class TfidfPublisherClassifier:
    """TF-IDF text features + top-N publisher one-hot -> logistic regression."""

    def __init__(self, max_features: int = 5_000, top_publishers: int = 15) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
        )
        self.publisher_encoder = PublisherOneHot(top_n=top_publishers)
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        self._calibrated: CalibratedClassifierCV | None = None
        self.decision_threshold: float = 0.5

    def _features(self, df: pd.DataFrame, *, fit: bool):
        text = _text_with_snippet(df)
        if fit:
            tf = self.vectorizer.fit_transform(text)
            pub = self.publisher_encoder.fit_transform(df["sources"])
        else:
            tf = self.vectorizer.transform(text)
            pub = self.publisher_encoder.transform(df["sources"])
        side = _side_matrix(df)
        return sparse.hstack(
            [tf, sparse.csr_matrix(pub), sparse.csr_matrix(side)],
            format="csr",
        )

    def fit(self, train_df: pd.DataFrame) -> "TfidfPublisherClassifier":
        X = self._features(train_df, fit=True)
        y = train_df["label_binary"].to_numpy(dtype=int)
        self.classifier.fit(X, y)
        log.info("Fitted TF-IDF+publisher classifier on %d rows (dim=%d)",
                 len(train_df), X.shape[1])
        return self

    def calibrate(self, val_df: pd.DataFrame) -> "TfidfPublisherClassifier":
        if val_df.empty or val_df["label_binary"].nunique() < 2:
            log.warning("Skipping calibration (val empty / single-class)")
            self._calibrated = None
            return self
        X_val = self._features(val_df, fit=False)
        y_val = val_df["label_binary"].to_numpy(dtype=int)
        raw_proba = self.classifier.predict_proba(X_val)[:, 1]
        cal = _make_prefit_calibrator(self.classifier, method="sigmoid")
        cal.fit(X_val, y_val)
        cal_proba = cal.predict_proba(X_val)[:, 1]
        if calibration_preserves_spread(raw_proba, cal_proba):
            self._calibrated = cal
            log.info(
                "Calibrated on %d val rows (raw_std=%.4f cal_std=%.4f).",
                len(val_df), float(raw_proba.std()), float(cal_proba.std()),
            )
        else:
            self._calibrated = None
            log.warning(
                "Rejected sigmoid calibration (raw_std=%.4f cal_std=%.4f); "
                "using uncalibrated classifier.",
                float(raw_proba.std()), float(cal_proba.std()),
            )
        return self

    def predict_proba_positive(self, df: pd.DataFrame) -> np.ndarray:
        X = self._features(df, fit=False)
        clf = self._calibrated if self._calibrated is not None else self.classifier
        return clf.predict_proba(X)[:, 1]

    def predict_binary(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba_positive(df)
        return (proba >= self.decision_threshold).astype(int)


def _pick_logreg_C(train_df: pd.DataFrame, val_df: pd.DataFrame,
                   max_features: int, top_publishers: int) -> float:
    if val_df.empty or val_df["label_binary"].nunique() < 2:
        return float(NLP_CONFIG.get("logreg_C", 1.0))
    best_C, best_auc = float(NLP_CONFIG.get("logreg_C", 1.0)), -1.0
    for C in (0.1, 0.5, 1.0, 2.0, 5.0):
        probe = TfidfPublisherClassifier(max_features=max_features, top_publishers=top_publishers)
        probe.classifier = LogisticRegression(
            C=C, max_iter=1_000, class_weight="balanced", random_state=42, solver="lbfgs",
        )
        probe.fit(train_df)
        probe.calibrate(val_df)
        proba = probe.predict_proba_positive(val_df)
        try:
            auc = roc_auc_score(val_df["label_binary"], proba)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_C = auc, C
    log.info("Selected LogReg C=%.1f (val AUC=%.3f)", best_C, best_auc)
    return best_C


def evaluate(model: TfidfPublisherClassifier, df: pd.DataFrame, split_name: str) -> dict:
    if df.empty:
        return {}
    proba = model.predict_proba_positive(df)
    preds = model.predict_binary(df)
    acc = accuracy_score(df["label_binary"], preds)
    try:
        auc = roc_auc_score(df["label_binary"], proba)
    except ValueError:
        auc = float("nan")
    log.info("%s - accuracy: %.3f  AUC: %.3f  (n=%d)",
             split_name, acc, auc, len(df))
    return {"accuracy": acc, "auc": auc, "n": len(df)}


def export_predictions(model: TfidfPublisherClassifier,
                       split_name: str, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba_positive(df)
    binary = model.predict_binary(df)
    conf = np.abs(proba - 0.5) * 2

    out = df[["ticker", "prediction_date"]].copy().reset_index(drop=True)
    out["split"] = split_name
    out["model_name"] = MODEL_NAME
    out["news_pred_proba"] = proba
    out["news_pred_binary"] = binary
    out["news_confidence"] = conf
    out["top_headlines"] = df["top_headlines"].reset_index(drop=True)
    out["actual_binary"] = df["label_binary"].reset_index(drop=True)
    out["model_version"] = getattr(model, "_model_version", None) or MODEL_VERSION
    return out


def load_tfidf_model(path: str | Path) -> TfidfPublisherClassifier:
    """Load ``nlp_baseline.joblib`` (dict payload or legacy classifier object)."""
    payload = joblib.load(path)
    if isinstance(payload, TfidfPublisherClassifier):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected TF-IDF model format in {path}: {type(payload)}")
    model = TfidfPublisherClassifier()
    model.vectorizer = payload["vectorizer"]
    model.publisher_encoder = payload["publisher_encoder"]
    model.classifier = payload["classifier"]
    model._calibrated = payload.get("calibrated")
    model._model_version = payload.get("model_version")
    model.decision_threshold = float(payload.get("decision_threshold", 0.5))
    return model


def save_predictions_csv(model: TfidfPublisherClassifier,
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
    log.info("Predictions saved to %s (%d rows)", out_path, len(combined))
    return out_path


def upsert_predictions_db(model: TfidfPublisherClassifier,
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
    parser = argparse.ArgumentParser(description="Train TF-IDF NLP baseline")
    parser.add_argument("--db", default=str(DATABASE_PATH))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-features", type=int,
                        default=int(NLP_CONFIG.get("max_features", 5000)))
    parser.add_argument("--top-publishers", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3],
                        help="Prediction horizon in trading days (default: 1)")
    parser.add_argument("--min-move-pct", type=float, default=0.0,
                        help="Drop TRAIN rows where |return_horizon| < this (eval keeps all).")
    parser.add_argument("--no-db-export", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    db_path = Path(args.db)
    csv_path = Path(args.output) if args.output else (PROCESSED_DATA_DIR / CSV_NAME)
    model_path = MODELS_DIR / MODEL_FILE
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TF-IDF + PUBLISHER ONE-HOT NLP BASELINE")
    print("=" * 70)
    print(f"DB            : {db_path}")
    print(f"Model output  : {model_path}")
    print(f"CSV output    : {csv_path}")
    print(f"Max features  : {args.max_features}   Top publishers: {args.top_publishers}")
    print()

    print("Step 1/4  Loading data ...")
    df = build_dataset(db_path, drop_rows_without_news=True, horizon=args.horizon)
    print(f"  {len(df):,} ticker-day rows (news-aligned, horizon={args.horizon})")

    print("Step 2/4  Splitting (chronological on news dates) ...")
    train_df, val_df, test_df = chronological_split(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    if args.min_move_pct > 0 and not train_df.empty:
        before = len(train_df)
        train_df = train_df[train_df["label_return"].abs() >= args.min_move_pct].copy()
        print(f"  Drop-flat filter (|ret|<{args.min_move_pct}%): "
              f"dropped {before - len(train_df)} train rows, kept {len(train_df)}")
    splits = {"train": train_df, "val": val_df, "test": test_df}

    print("Step 3/4  Training ...")
    best_C = _pick_logreg_C(train_df, val_df, args.max_features, args.top_publishers)
    model = TfidfPublisherClassifier(
        max_features=args.max_features,
        top_publishers=args.top_publishers,
    )
    model.classifier = LogisticRegression(
        C=best_C, max_iter=1_000, class_weight="balanced", random_state=42, solver="lbfgs",
    )
    model.fit(train_df)
    if not args.no_calibration:
        model.calibrate(val_df)
    if not val_df.empty and val_df["label_binary"].nunique() >= 2:
        model.decision_threshold = tune_threshold_balanced_accuracy(
            val_df["label_binary"].to_numpy(),
            model.predict_proba_positive(val_df),
        )
        log.info("TF-IDF decision threshold=%.3f", model.decision_threshold)

    print()
    metrics = {name: evaluate(model, split_df, name) for name, split_df in splits.items()}

    print("\nStep 4/4  Saving outputs ...")
    payload = {
        "vectorizer": model.vectorizer,
        "publisher_encoder": model.publisher_encoder,
        "classifier": model.classifier,
        "calibrated": model._calibrated,
        "decision_threshold": model.decision_threshold,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
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

    from src.ml.model_diagnostics import per_ticker_auc, print_per_ticker_auc  # noqa: E402
    test_df = splits.get("test")
    if test_df is not None and not test_df.empty:
        scored = test_df.copy()
        scored["news_pred_proba"] = model.predict_proba_positive(test_df)
        scored["actual_binary"] = scored["label_binary"]
        tt = per_ticker_auc(scored, proba_col="news_pred_proba", min_rows=10)
        print_per_ticker_auc("Per-ticker test AUC (TF-IDF)", tt)
    print()
    print(f"  model_name    : {MODEL_NAME}")
    print(f"  model_version : {MODEL_VERSION}")
    print("=" * 70)


if __name__ == "__main__":
    main()
