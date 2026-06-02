#!/usr/bin/env python3
"""Evaluate every model head (+ simple baselines) on the held-out test split.

Inputs
------
  data/processed/final_ensemble_predictions.csv
  data/database.db (labels table)               (for the previous_day_direction baseline)

Outputs (written to data/processed/)
-----------------------------------
  evaluation_overall.csv          - metrics for all rows
  evaluation_by_ticker.csv        - per-ticker breakdown
  evaluation_summary.txt          - human-readable table (all / has_news / high_conf)
  evaluation_by_confidence.csv    - accuracy per confidence bucket

Reported models:
  - lstm_price
  - news_tfidf
  - news_embeddings
  - ensemble
  - always_up
  - previous_day_direction

Metrics: accuracy, precision_up, recall_up, f1_up, auc (when both classes
present), n.

Subset breakdown (printed in summary, also written to evaluation_overall.csv):
  - all        : every labeled test row (current behavior)
  - has_news   : rows where has_news == 1 (real news, not 0.5 fill)
  - high_conf  : rows where ensemble_confidence >= 0.3

LSTM diagnostics:
  Reads price_predictions.csv (produced by train_lstm.py) and prints the
  pre-calibration proba distribution to flag model collapse early.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import joblib
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

from src.config import DATABASE_PATH, MODELS_DIR, PROCESSED_DATA_DIR  # noqa: E402


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


def _load_lstm_decision_threshold() -> float:
    """Read the tuned cutoff saved by train_lstm.py (defaults to 0.5)."""
    path = MODELS_DIR / "lstm_model.pt"
    if not path.exists():
        return 0.5
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return float(ckpt.get("decision_threshold", 0.5))
    except Exception:
        return 0.5


def _load_news_decision_thresholds() -> dict[str, float]:
    """Read tuned cutoffs saved by train_nlp / train_news_embeddings."""
    out: dict[str, float] = {"news_tfidf": 0.5, "news_embeddings": 0.5}
    for name, fname in (
        ("news_tfidf", "news_tfidf.joblib"),
        ("news_embeddings", "news_embeddings.joblib"),
    ):
        path = MODELS_DIR / fname
        if not path.exists():
            continue
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict):
                out[name] = float(obj.get("decision_threshold", 0.5))
            elif hasattr(obj, "decision_threshold"):
                out[name] = float(obj.decision_threshold)
        except Exception:
            pass
    return out


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

    # Build subset masks for diagnostic evaluation.
    has_news_mask = None
    if "has_news" in df.columns:
        has_news_mask = df["has_news"].fillna(0).astype(int).to_numpy().astype(bool)
    high_conf_mask = None
    if "ensemble_confidence" in df.columns:
        high_conf_mask = (df["ensemble_confidence"].fillna(0.0) >= 0.3).to_numpy()
    elif "ensemble_pred_proba" in df.columns:
        conf = (df["ensemble_pred_proba"].fillna(0.5) - 0.5).abs() * 2
        high_conf_mask = (conf >= 0.3).to_numpy()

    news_scored_mask = None
    news_oos_mask = None
    if "news_tfidf_split" in df.columns and "split" in df.columns:
        news_scored_mask = (
            df["news_tfidf_split"].fillna("") == df["split"]
        ).to_numpy()
        news_oos_mask = (
            df["news_tfidf_split"].fillna("").isin(["val", "test"])
            & (df["news_tfidf_split"].fillna("") == df["split"])
        ).to_numpy()

    y_true = df["actual_binary"].to_numpy()
    lstm_threshold = _load_lstm_decision_threshold()
    news_thresholds = _load_news_decision_thresholds()
    overall_rows: list[dict] = []
    by_ticker_rows: list[dict] = []

    def _add_subset(name: str, y_pred: np.ndarray, y_proba: Optional[np.ndarray],
                    subset: str = "all", mask: Optional[np.ndarray] = None):
        if mask is not None:
            if mask.sum() == 0:
                return
            yt = y_true[mask]
            yp = y_pred[mask]
            ya = y_proba[mask] if y_proba is not None else None
        else:
            yt, yp, ya = y_true, y_pred, y_proba
        metrics = _compute_metrics(yt, yp, ya)
        overall_rows.append({"model": name, "split": split, "subset": subset, **metrics})

    def _add(name: str, y_pred: np.ndarray, y_proba: Optional[np.ndarray]):
        _add_subset(name, y_pred, y_proba, subset="all")
        if has_news_mask is not None:
            _add_subset(name, y_pred, y_proba, subset="has_news", mask=has_news_mask)
        if news_scored_mask is not None and news_scored_mask.any():
            _add_subset(name, y_pred, y_proba, subset="news_scored", mask=news_scored_mask)
        if news_oos_mask is not None and news_oos_mask.any():
            _add_subset(name, y_pred, y_proba, subset="news_oos", mask=news_oos_mask)
        if high_conf_mask is not None:
            _add_subset(name, y_pred, y_proba, subset="high_conf", mask=high_conf_mask)
        if "high_vol" in df.columns:
            low_vol = (df["high_vol"].fillna(0).astype(float) == 0).to_numpy()
            high_vol = (df["high_vol"].fillna(0).astype(float) == 1).to_numpy()
            if low_vol.any():
                _add_subset(name, y_pred, y_proba, subset="low_vol", mask=low_vol)
            if high_vol.any():
                _add_subset(name, y_pred, y_proba, subset="high_vol", mask=high_vol)

        # Per-ticker (all rows)
        for ticker, sub in df.groupby("ticker"):
            mask = (df["ticker"] == ticker).to_numpy()
            sub_true = y_true[mask]
            sub_pred = y_pred[mask]
            sub_proba = y_proba[mask] if y_proba is not None else None
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
        threshold = lstm_threshold if name == "lstm_price" else news_thresholds.get(name, 0.5)
        pred = (proba >= threshold).astype(int)
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

    overall = pd.DataFrame(overall_rows).sort_values(
        ["subset", "accuracy"], ascending=[True, False]
    )
    by_ticker = pd.DataFrame(by_ticker_rows).sort_values(["model", "ticker"])
    return overall, by_ticker, overall_rows


def _lstm_diagnostics(processed_dir: Path) -> str:
    """Read price_predictions.csv and report LSTM proba distribution statistics.

    This catches model collapse (all probas near 0.5, zero DOWN predictions)
    before the full ensemble evaluation runs.
    """
    csv = processed_dir / "price_predictions.csv"
    if not csv.exists():
        return "  [skip] price_predictions.csv not found"
    df = pd.read_csv(csv)
    test = df[df["split"] == "test"].copy() if "split" in df.columns else df
    if test.empty or "financial_pred_proba" not in test.columns:
        return "  [skip] no test rows or missing column"

    proba = test["financial_pred_proba"].astype(float)
    threshold = _load_lstm_decision_threshold()
    pred = (proba >= threshold).astype(int)
    y_true = test["actual_binary"].dropna().astype(int) if "actual_binary" in test.columns else None
    n = len(proba)
    n_up = int((pred == 1).sum())
    up_pct = n_up / max(n, 1)
    lines = [
        f"  n_test={n}   threshold={threshold:.3f}   "
        f"UP_predictions={n_up} ({up_pct:.0%})  "
        f"DOWN_predictions={n - n_up} ({(n-n_up)/max(n,1):.0%})",
        f"  proba  min={proba.min():.4f}  max={proba.max():.4f}  "
        f"mean={proba.mean():.4f}  std={proba.std():.4f}",
    ]
    if proba.std() < 0.01:
        lines.append("  *** COLLAPSED: std < 0.01 — model is essentially constant ***")
    if n_up == n:
        lines.append("  *** DEGENERATE: 100% UP predictions — balanced accuracy = 50% ***")
    if y_true is not None and len(y_true) == n and len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, proba))
            lines.append(f"  test AUC={auc:.3f}")
            if up_pct > 0.90 and auc < 0.52:
                lines.append(
                    "  *** COLLAPSE WARNING: UP%>90% and AUC<0.52 — "
                    "ensemble will downweight LSTM ***"
                )
        except ValueError:
            pass
    return "\n".join(lines)


def _volatility_diagnostics(processed_dir: Path) -> str:
    """Report expected-move model skill on the test split."""
    csv = processed_dir / "volatility_predictions.csv"
    if not csv.exists():
        return "  [skip] volatility_predictions.csv not found"
    df = pd.read_csv(csv)
    test = df[(df["split"] == "test") & df["actual_abs_return_pct"].notna()]
    if test.empty:
        return "  [skip] no labeled volatility test rows"
    pred = test["expected_move_pct"].astype(float)
    actual = test["actual_abs_return_pct"].astype(float)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(actual, pred)
    med = float(actual.median())
    y_high = (actual > med).astype(int)
    try:
        auc = float(roc_auc_score(y_high, pred))
    except ValueError:
        auc = float("nan")
    return (
        f"  n_test={len(test)}  MAE={mae:.3f}%  high-move AUC={auc:.3f}  "
        f"(median |return|={med:.3f}%)"
    )


def _data_coverage_report(db_path: Path) -> str:
    """Print news vs price date range per ticker to quantify news sparsity."""
    if not db_path.exists():
        return "  [skip] database not found"
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        prices = pd.read_sql_query(
            "SELECT ticker, MIN(date) AS price_min, MAX(date) AS price_max, "
            "COUNT(*) AS price_rows FROM prices WHERE ticker != 'SPY' "
            "GROUP BY ticker ORDER BY ticker",
            conn,
        )
        news = pd.read_sql_query(
            "SELECT ticker, MIN(published_at) AS news_min, MAX(published_at) AS news_max, "
            "COUNT(*) AS news_rows FROM news GROUP BY ticker ORDER BY ticker",
            conn,
        )
    finally:
        conn.close()

    if prices.empty:
        return "  [skip] no price data in DB"

    merged = prices.merge(news, on="ticker", how="left")
    lines = [
        f"  {'Ticker':6s}  {'Price from':10s}  {'Price to':10s}  "
        f"{'News from':10s}  {'News to':10s}  {'News rows':>9s}  {'Price rows':>10s}"
    ]
    for _, r in merged.iterrows():
        news_min = r.get("news_min", "—") or "—"
        news_max = r.get("news_max", "—") or "—"
        news_rows = int(r["news_rows"]) if pd.notna(r.get("news_rows")) else 0
        lines.append(
            f"  {r['ticker']:6s}  {str(r['price_min'])[:10]:10s}  "
            f"{str(r['price_max'])[:10]:10s}  {str(news_min)[:10]:10s}  "
            f"{str(news_max)[:10]:10s}  {news_rows:>9,d}  {int(r['price_rows']):>10,d}"
        )
    return "\n".join(lines)


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
    subsets = ["all", "has_news", "news_scored", "news_oos", "high_conf", "low_vol", "high_vol"] if "subset" in overall.columns else ["all"]
    lines = []
    for subset in subsets:
        sub = overall[overall["subset"] == subset] if "subset" in overall.columns else overall
        if sub.empty:
            continue
        n_label = f"  subset={subset!r}" if subset != "all" else ""
        lines += [
            "=" * 76,
            f"EVALUATION SUMMARY (split={split}){n_label}",
            "=" * 76,
            "",
            f"{'model':<28s}{'acc':>8s}{'prec_up':>10s}{'recall_up':>12s}"
            f"{'f1_up':>10s}{'auc':>8s}{'n':>8s}",
            "-" * 76,
        ]
        for _, r in sub.iterrows():
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
        lines.append("=" * 76)
        lines.append("")
    return "\n".join(lines).rstrip()


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

    print("\n--- LSTM Probability Diagnostics ---")
    print(_lstm_diagnostics(output_dir))

    print("\n--- Volatility / Expected-Move Diagnostics ---")
    print(_volatility_diagnostics(output_dir))

    print("\n--- Data Coverage Report ---")
    print(_data_coverage_report(DATABASE_PATH))

    summary = _format_summary(overall, args.split)
    summary_path.write_text(summary + "\n")

    conviction = _evaluate_by_confidence(
        predictions_path, split=args.split, n_buckets=args.n_buckets,
    )
    if conviction is not None:
        conviction.to_csv(conviction_path, index=False)

    print("\n" + summary)
    print(f"\nSaved:\n  {overall_path}\n  {by_ticker_path}\n  {summary_path}")
    if conviction is not None:
        print(f"  {conviction_path}")


if __name__ == "__main__":
    main()
