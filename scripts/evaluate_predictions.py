#!/usr/bin/env python3
"""Evaluate every model head (+ simple baselines) on the held-out test split.

Inputs
------
  data/processed/final_ensemble_predictions.csv
  data/database.db (labels table)               (for the previous_day_direction baseline)

Outputs (written to data/processed/)
-----------------------------------
  evaluation_overall.csv
  evaluation_by_ticker.csv
  evaluation_summary.txt

Reported models:
  - lstm_price
  - news_tfidf
  - news_embeddings
  - ensemble
  - always_up
  - previous_day_direction

Metrics: accuracy, precision_up, recall_up, f1_up, auc (when both classes
present), n.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR  # noqa: E402


MODELS = [
    ("lstm_price", "financial_pred_proba"),
    ("news_tfidf", "news_tfidf_pred_proba"),
    ("news_embeddings", "news_embeddings_pred_proba"),
    ("ensemble", "ensemble_pred_proba"),
]


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    out = {
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "precision_up":  float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_up":     float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_up":         float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "n":             int(len(y_true)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")
    return out


def _baseline_always_up(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return np.ones(len(df), dtype=int), np.full(len(df), 0.5, dtype=float)


def _label_column(horizon: int) -> str:
    if horizon == 1:
        return "label_binary"
    if horizon == 3:
        return "label_binary_h3"
    raise ValueError(f"Unsupported horizon {horizon}")


def _load_labels_from_db(db_path: Path, horizon: int = 1) -> pd.DataFrame:
    """Load the labels table, using the horizon-appropriate column."""
    import sqlite3

    col = _label_column(horizon)
    con = sqlite3.connect(str(db_path))
    try:
        labels = pd.read_sql_query(
            f"SELECT ticker, date AS label_date, {col} AS direction "
            f"FROM labels WHERE {col} IS NOT NULL",
            con,
        )
    finally:
        con.close()
    return labels


def _baseline_previous_day_direction(
    df: pd.DataFrame, labels: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Predict next-day direction = today's direction for each ticker."""
    labels = labels.sort_values(["ticker", "label_date"]).copy()
    labels["prev_direction"] = labels.groupby("ticker")["direction"].shift(1)

    labels = labels.rename(columns={"label_date": "prediction_date"})

    joined = df.merge(
        labels[["ticker", "prediction_date", "prev_direction"]],
        on=["ticker", "prediction_date"],
        how="left",
    )

    y_pred = joined["prev_direction"].fillna(1).astype(int).to_numpy()
    y_proba = np.where(y_pred == 1, 0.75, 0.25)
    return y_pred, y_proba


def evaluate_all(
    predictions_path: Path,
    labels_path: Optional[Path] = None,
    split: str = "test",
    horizon: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    df = pd.read_csv(predictions_path)
    if "split" not in df.columns:
        raise ValueError(f"{predictions_path} is missing `split` column")
    df = df[(df["split"] == split) & df["actual_binary"].notna()].copy()
    if len(df) == 0:
        raise RuntimeError(f"No labeled rows found for split={split!r}")

    df["actual_binary"] = df["actual_binary"].astype(int)
    y_true = df["actual_binary"].to_numpy()

    overall_rows: list[dict] = []
    by_ticker_rows: list[dict] = []

    def _add(name: str, y_pred: np.ndarray, y_proba: Optional[np.ndarray]):
        metrics = _compute_metrics(y_true, y_pred, y_proba)
        overall_rows.append({"model": name, "split": split, **metrics})

        for ticker, sub in df.groupby("ticker"):
            mask = df["ticker"] == ticker
            sub_true = sub["actual_binary"].to_numpy()
            sub_pred = y_pred[mask.to_numpy()]
            sub_proba = y_proba[mask.to_numpy()] if y_proba is not None else None
            m = _compute_metrics(sub_true, sub_pred, sub_proba)
            by_ticker_rows.append({
                "model": name,
                "ticker": ticker,
                "split": split,
                **m,
            })

    # Model heads
    for name, col in MODELS:
        if col not in df.columns:
            print(f"  [skip] {name}: column {col!r} not in predictions")
            continue
        proba = df[col].astype(float).to_numpy()
        pred = (proba >= 0.5).astype(int)
        _add(name, pred, proba)

    # Baselines
    pred, proba = _baseline_always_up(df)
    _add("always_up", pred, proba)

    labels: Optional[pd.DataFrame] = None
    if labels_path is not None and labels_path.exists():
        labels = pd.read_csv(labels_path)
    elif DATABASE_PATH.exists():
        try:
            labels = _load_labels_from_db(DATABASE_PATH, horizon=horizon)
        except Exception as exc:
            print(f"  [skip] previous_day_direction: could not read labels from DB ({exc})")

    if labels is not None and len(labels) > 0:
        pred, proba = _baseline_previous_day_direction(df, labels)
        _add("previous_day_direction", pred, proba)
    else:
        print("  [skip] previous_day_direction: no labels available")

    overall = pd.DataFrame(overall_rows).sort_values("accuracy", ascending=False)
    by_ticker = pd.DataFrame(by_ticker_rows).sort_values(["model", "ticker"])
    return overall, by_ticker, overall_rows


def _evaluate_by_confidence(
    predictions_path: Path,
    split: str = "test",
    n_buckets: int = 5,
) -> Optional[pd.DataFrame]:
    """Return a DataFrame with accuracy per confidence bucket for each head.

    `confidence` is recomputed from probability as |p - 0.5| * 2 so that we
    get a consistent bucketing regardless of the upstream calibration.
    """
    df = pd.read_csv(predictions_path)
    if "split" not in df.columns:
        return None
    df = df[(df["split"] == split) & df["actual_binary"].notna()].copy()
    if df.empty:
        return None

    df["actual_binary"] = df["actual_binary"].astype(int)

    rows: list[dict] = []
    for name, col in MODELS:
        if col not in df.columns:
            continue
        proba = df[col].astype(float).to_numpy()
        conf = np.abs(proba - 0.5) * 2
        pred = (proba >= 0.5).astype(int)
        correct = (pred == df["actual_binary"].to_numpy()).astype(int)
        # Skip buckets if everyone collapses to 0 confidence
        try:
            buckets = pd.qcut(conf, q=n_buckets, duplicates="drop")
        except ValueError:
            buckets = pd.cut(conf, bins=n_buckets)
        for bucket, idx in pd.Series(correct).groupby(buckets, observed=True):
            rows.append({
                "model": name,
                "bucket": str(bucket),
                "conf_min": float(conf[idx.index].min()),
                "conf_max": float(conf[idx.index].max()),
                "accuracy": float(idx.mean()),
                "n": int(len(idx)),
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _format_summary(overall: pd.DataFrame, split: str) -> str:
    lines = [
        "=" * 72,
        f"EVALUATION SUMMARY (split={split})",
        "=" * 72,
        "",
        f"{'model':<28s}{'acc':>8s}{'prec_up':>10s}{'recall_up':>12s}"
        f"{'f1_up':>10s}{'auc':>8s}{'n':>8s}",
        "-" * 72,
    ]
    for _, r in overall.iterrows():
        auc = "nan" if (isinstance(r["auc"], float) and np.isnan(r["auc"])) else f"{r['auc']:.3f}"
        lines.append(
            f"{r['model']:<28s}"
            f"{r['accuracy']:>8.3f}"
            f"{r['precision_up']:>10.3f}"
            f"{r['recall_up']:>12.3f}"
            f"{r['f1_up']:>10.3f}"
            f"{auc:>8s}"
            f"{int(r['n']):>8d}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all models on held-out split")
    parser.add_argument(
        "--input", default=None,
        help=f"Predictions CSV (default: {PROCESSED_DATA_DIR / 'final_ensemble_predictions.csv'})",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Optional labels CSV; if omitted, labels are loaded from the SQLite DB.",
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Where to write evaluation_*.csv/txt (default: {PROCESSED_DATA_DIR})",
    )
    parser.add_argument(
        "--horizon", type=int, default=1, choices=[1, 3],
        help="Prediction horizon for the previous-day baseline (default: 1)",
    )
    parser.add_argument(
        "--n-buckets", type=int, default=5,
        help="Number of confidence deciles for evaluation_by_confidence.csv",
    )
    args = parser.parse_args()

    predictions_path = Path(args.input) if args.input else PROCESSED_DATA_DIR / "final_ensemble_predictions.csv"
    labels_path = Path(args.labels) if args.labels else None
    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EVALUATE PREDICTIONS")
    print("=" * 70)
    print(f"Input   : {predictions_path}")
    print(f"Labels  : {labels_path if labels_path else DATABASE_PATH}")
    print(f"Split   : {args.split}")

    overall, by_ticker, _ = evaluate_all(
        predictions_path, labels_path, split=args.split, horizon=args.horizon,
    )

    overall_path = output_dir / "evaluation_overall.csv"
    by_ticker_path = output_dir / "evaluation_by_ticker.csv"
    summary_path = output_dir / "evaluation_summary.txt"
    conviction_path = output_dir / "evaluation_by_confidence.csv"

    overall.to_csv(overall_path, index=False)
    by_ticker.to_csv(by_ticker_path, index=False)
    summary = _format_summary(overall, args.split)
    summary_path.write_text(summary + "\n")

    conviction = _evaluate_by_confidence(
        predictions_path, split=args.split, n_buckets=args.n_buckets,
    )
    if conviction is not None:
        conviction.to_csv(conviction_path, index=False)

    print(summary)
    print(f"\nSaved:\n  {overall_path}\n  {by_ticker_path}\n  {summary_path}")
    if conviction is not None:
        print(f"  {conviction_path}")


if __name__ == "__main__":
    main()
