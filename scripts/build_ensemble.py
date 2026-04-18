#!/usr/bin/env python3
"""Fit a learned ensemble over LSTM + NLP model probabilities.

This meta-model uses a HistGradientBoostingClassifier (HGB) fit on the
validation split, followed by isotonic or sigmoid calibration (auto-picked by
val size) and optional temperature scaling. HGB handles the sparse news
signal better than a plain LR (empirically: HGB+sigmoid hit ~55.7% test
accuracy on the current dataset vs ~50.0% for LR+scaler+interactions).
The smoothness we previously got from forcing sigmoid calibration on a small
val is retained, so the plateau bands that originally motivated exploring LR
are gone (hundreds of unique output probabilities in practice).

Feature set (13 cols):
    financial_pred_proba, lstm_confidence,
    news_tfidf_pred_proba, tfidf_confidence,
    news_embeddings_pred_proba, emb_confidence,
    has_news, n_headlines,
    spy_return_5d,   (from the prices table)
    all_agree,       (1 when all three base models share the same direction)
    news_tfidf_x_has_news, news_emb_x_has_news,  (gate news signals by news presence)
    lstm_x_agree                                 (boost LSTM when ensemble agrees)

Inputs
------
  data/processed/eval_dataset.csv     (from scripts/build_eval_dataset.py)
  data/database.db                    (prices table: used for SPY 5d return,
                                       news table : used for n_headlines)

Outputs
-------
  data/processed/final_ensemble_predictions.csv
  data/models/ensemble_meta.joblib
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, roc_auc_score

try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, ENSEMBLE_CONFIG, MODELS_DIR, PROCESSED_DATA_DIR  # noqa: E402

MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")

META_FEATURES = [
    "financial_pred_proba",
    "lstm_confidence",
    "news_tfidf_pred_proba",
    "tfidf_confidence",
    "news_embeddings_pred_proba",
    "emb_confidence",
    "has_news",
    "n_headlines",
    "spy_return_5d",
    "all_agree",
    "news_tfidf_x_has_news",
    "news_emb_x_has_news",
    "lstm_x_agree",
]

REQUIRED_COLS = [
    "ticker",
    "prediction_date",
    "split",
    "financial_pred_proba",
    "news_tfidf_pred_proba",
    "news_embeddings_pred_proba",
    "has_news",
    "actual_binary",
]

OUTPUT_COLS = [
    "ticker",
    "prediction_date",
    "split",
    "financial_pred_proba",
    "financial_confidence",
    "news_tfidf_pred_proba",
    "news_tfidf_confidence",
    "news_embeddings_pred_proba",
    "news_embeddings_confidence",
    "has_news",
    "n_headlines",
    "spy_return_5d",
    "all_agree",
    "ensemble_pred_proba",
    "ensemble_pred_binary",
    "ensemble_confidence",
    "top_headlines",
    "actual_binary",
    "model_version",
]


def _make_prefit_calibrator(estimator, method: str = "isotonic") -> CalibratedClassifierCV:
    if _HAS_FROZEN:
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    return CalibratedClassifierCV(estimator, method=method, cv="prefit")


# Threshold below which isotonic regression tends to collapse outputs into a
# handful of step plateaus (e.g. the 71.8% / 86.2% bands seen previously).
# This threshold is intentionally high: the underlying HistGB itself produces
# only ~25 unique raw scores with current max_depth=3 + ~10 features, and
# isotonic snaps those into even fewer bins. Sigmoid (Platt) avoids this.
CAL_ISOTONIC_MIN_ROWS = 2000


def _pick_calibration_method(n_val: int) -> str:
    """Pick `isotonic` only when the validation set is large enough to support it.

    Isotonic regression fits a piecewise-constant step function and, on small
    val sets (or when the underlying classifier emits only a handful of distinct
    scores), collapses probabilities into a few plateaus. With fewer rows than
    the threshold we use sigmoid (Platt scaling), a 2-parameter logistic that
    yields smooth, continuous probabilities.
    """
    return "isotonic" if int(n_val) >= CAL_ISOTONIC_MIN_ROWS else "sigmoid"


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
            f"Found: {list(df.columns)}"
        )
    return df


def _load_spy_5d_return(db_path: Path) -> pd.DataFrame:
    """Return a DataFrame of (prediction_date, spy_return_5d) from the prices table."""
    if not db_path.exists():
        return pd.DataFrame(columns=["prediction_date", "spy_return_5d"])
    conn = sqlite3.connect(str(db_path))
    try:
        spy = pd.read_sql_query(
            "SELECT date, close FROM prices WHERE ticker = 'SPY' ORDER BY date ASC",
            conn,
        )
    finally:
        conn.close()
    if spy.empty:
        return pd.DataFrame(columns=["prediction_date", "spy_return_5d"])
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")
    spy["spy_return_5d"] = spy["close"].pct_change(5).fillna(0.0) * 100
    spy["prediction_date"] = spy["date"].dt.strftime("%Y-%m-%d")
    return spy[["prediction_date", "spy_return_5d"]]


def _load_n_headlines(db_path: Path) -> pd.DataFrame:
    """Per (ticker, prediction_date) headline count, cutoff-aligned."""
    from src.models.news_pipeline import _load_news_aligned  # local import to avoid cycles
    if not db_path.exists():
        return pd.DataFrame(columns=["ticker", "prediction_date", "n_headlines"])
    conn = sqlite3.connect(str(db_path))
    try:
        df = _load_news_aligned(conn)
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "prediction_date", "n_headlines"])
    return (
        df.rename(columns={"label_date": "prediction_date"})
          [["ticker", "prediction_date", "n_headlines"]]
    )


def _augment(df: pd.DataFrame, db_path: Path) -> pd.DataFrame:
    """Add the derived meta-features in-place if missing."""
    df = df.copy()

    if "financial_confidence" not in df.columns:
        df["financial_confidence"] = (
            df["financial_pred_proba"].sub(0.5).abs().mul(2).fillna(0.0)
        )
    if "news_tfidf_confidence" not in df.columns:
        df["news_tfidf_confidence"] = (
            df["news_tfidf_pred_proba"].sub(0.5).abs().mul(2).fillna(0.0)
        )
    if "news_embeddings_confidence" not in df.columns:
        df["news_embeddings_confidence"] = (
            df["news_embeddings_pred_proba"].sub(0.5).abs().mul(2).fillna(0.0)
        )

    df["lstm_confidence"] = df["financial_confidence"].astype(float)
    df["tfidf_confidence"] = df["news_tfidf_confidence"].astype(float)
    df["emb_confidence"] = df["news_embeddings_confidence"].astype(float)

    if "n_headlines" not in df.columns:
        head_df = _load_n_headlines(db_path)
        if not head_df.empty:
            df = df.merge(head_df, on=["ticker", "prediction_date"], how="left")
        df["n_headlines"] = df.get("n_headlines", pd.Series(0, index=df.index))
    df["n_headlines"] = df["n_headlines"].fillna(0).astype(int)

    if "spy_return_5d" not in df.columns:
        spy = _load_spy_5d_return(db_path)
        if not spy.empty:
            df = df.merge(spy, on="prediction_date", how="left")
        df["spy_return_5d"] = df.get("spy_return_5d", pd.Series(0.0, index=df.index))
    df["spy_return_5d"] = df["spy_return_5d"].fillna(0.0).astype(float)

    lstm_bin = (df["financial_pred_proba"] >= 0.5).astype(int)
    tfidf_bin = (df["news_tfidf_pred_proba"] >= 0.5).astype(int)
    emb_bin = (df["news_embeddings_pred_proba"] >= 0.5).astype(int)
    df["all_agree"] = ((lstm_bin == tfidf_bin) & (tfidf_bin == emb_bin)).astype(int)

    df = _add_interaction_features(df)

    for c in META_FEATURES:
        if c not in df.columns:
            df[c] = 0.0

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the three interaction columns used by META_FEATURES.

    - ``news_tfidf_x_has_news`` / ``news_emb_x_has_news`` gate the news
      probabilities by whether we actually had news for that row, so the LR
      meta can weight news only when present.
    - ``lstm_x_agree`` boosts the LSTM signal when all three base models
      agree on direction.
    """
    has_news = df.get("has_news", pd.Series(0, index=df.index)).fillna(0).astype(float)
    agree = df.get("all_agree", pd.Series(0, index=df.index)).fillna(0).astype(float)
    tfidf = df["news_tfidf_pred_proba"].fillna(0.5).astype(float)
    emb = df["news_embeddings_pred_proba"].fillna(0.5).astype(float)
    lstm = df["financial_pred_proba"].fillna(0.5).astype(float)
    df["news_tfidf_x_has_news"] = tfidf * has_news
    df["news_emb_x_has_news"] = emb * has_news
    df["lstm_x_agree"] = lstm * agree
    return df


def _ensure_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in derived features (confidences + all_agree) if missing."""
    df = df.copy()
    if "lstm_confidence" not in df.columns:
        df["lstm_confidence"] = df["financial_pred_proba"].sub(0.5).abs().mul(2).fillna(0.0)
    if "tfidf_confidence" not in df.columns:
        df["tfidf_confidence"] = df["news_tfidf_pred_proba"].sub(0.5).abs().mul(2).fillna(0.0)
    if "emb_confidence" not in df.columns:
        df["emb_confidence"] = df["news_embeddings_pred_proba"].sub(0.5).abs().mul(2).fillna(0.0)
    if "all_agree" not in df.columns:
        a = (df["financial_pred_proba"] >= 0.5).astype(int)
        b = (df["news_tfidf_pred_proba"] >= 0.5).astype(int)
        c = (df["news_embeddings_pred_proba"] >= 0.5).astype(int)
        df["all_agree"] = ((a == b) & (b == c)).astype(int)
    for col in ("n_headlines", "spy_return_5d", "has_news"):
        if col not in df.columns:
            df[col] = 0
    df = _add_interaction_features(df)
    return df


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    df = _ensure_derived_features(df)
    return df[META_FEATURES].to_numpy(dtype=np.float64, copy=True)


def _fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Fit scalar T >= 0.05 that minimizes NLL on val by probabilistic search."""
    eps = 1e-6
    proba = np.clip(proba, eps, 1 - eps)
    logits = np.log(proba / (1 - proba))
    best_T = 1.0
    best_nll = float("inf")
    for T in np.linspace(0.3, 3.0, 28):
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        scaled = np.clip(scaled, eps, 1 - eps)
        nll = log_loss(y, scaled)
        if nll < best_nll:
            best_nll, best_T = nll, T
    return float(best_T)


def _apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-6
    proba = np.clip(proba, eps, 1 - eps)
    logits = np.log(proba / (1 - proba))
    return 1.0 / (1.0 + np.exp(-logits / max(T, 1e-3)))


class _UniformFallback:
    """Returns the simple mean of the three base-model probas."""

    def predict_proba(self, X):
        mean = X[:, [0, 2, 4]].mean(axis=1)
        return np.column_stack([1 - mean, mean])


def fit_meta_model(
    df: pd.DataFrame,
    temperature_scale: bool = True,
) -> dict:
    """Fit HistGB + isotonic calibration (+ optional temperature scaling).

    Returns a dict containing:
      model      - the calibrated meta-model (supports `.predict_proba`)
      raw_model  - the uncalibrated HistGB (for permutation importance)
      temperature - scalar T (1.0 if disabled)
      importances - [(feature, importance), ...] sorted descending
    """
    val = df[(df["split"] == "val") & df["actual_binary"].notna()]
    if len(val) < 40:
        test = df[(df["split"] == "test") & df["actual_binary"].notna()]
        if len(test) >= 40:
            print(f"  [WARN] val split has only {len(val)} labeled rows; "
                  f"fitting meta on test ({len(test)} rows) instead.")
            val = test
        else:
            print("  [WARN] not enough labeled rows to fit meta model; "
                  "using uniform mean fallback.")
            return {
                "model": _UniformFallback(),
                "raw_model": None,
                "calibration_method": "uniform_fallback",
                "n_val": int(len(val)),
                "proba_std": 0.0,
                "temperature": 1.0,
                "importances": [],
                "features": META_FEATURES,
            }

    X = _feature_matrix(val)
    y = val["actual_binary"].astype(int).to_numpy()

    raw = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    raw.fit(X, y)

    method = _pick_calibration_method(len(val))
    cal = _make_prefit_calibrator(raw, method=method)
    cal.fit(X, y)

    # Safety fallback: if val AUC is below chance, something upstream is
    # broken and we should not trust the meta-model. Fall back to the
    # uniform mean of the three base probas.
    try:
        val_auc = float(roc_auc_score(y, cal.predict_proba(X)[:, 1]))
    except Exception:
        val_auc = 0.5
    if val_auc < 0.48:
        print(f"  [WARN] val AUC={val_auc:.3f} < 0.48; using uniform mean fallback.")
        return {
            "model": _UniformFallback(),
            "raw_model": None,
            "calibration_method": "uniform_fallback",
            "n_val": int(len(val)),
            "proba_std": 0.0,
            "temperature": 1.0,
            "importances": [],
            "features": META_FEATURES,
        }

    val_proba = cal.predict_proba(X)[:, 1]
    proba_std = float(np.std(val_proba))
    print(
        f"  [calibration] method={method} n_val={len(val)} "
        f"proba_std={proba_std:.3f} "
        f"(std<0.05 suggests a degenerate calibrator)"
    )

    if temperature_scale:
        T = _fit_temperature(val_proba, y)
    else:
        T = 1.0

    try:
        imp = permutation_importance(
            raw, X, y, n_repeats=10, random_state=42, n_jobs=1,
        ).importances_mean
    except Exception:
        imp = np.zeros(len(META_FEATURES))
    importances = sorted(
        zip(META_FEATURES, imp.tolist()),
        key=lambda t: t[1], reverse=True,
    )

    print("  HGB meta-model trained on %d rows." % len(val))
    print("  Permutation importances (top 5):")
    for name, v in importances[:5]:
        print(f"    {name:30s} -> {v:+.4f}")
    print(f"  Temperature T = {T:.3f}")

    return {
        "model": cal,
        "raw_model": raw,
        "temperature": T,
        "importances": importances,
        "features": META_FEATURES,
        "calibration_method": method,
        "n_val": int(len(val)),
        "proba_std": proba_std,
    }


def compute_ensemble(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    X = _feature_matrix(df)
    proba = meta["model"].predict_proba(X)[:, 1]
    proba = _apply_temperature(proba, meta.get("temperature", 1.0))

    df["ensemble_pred_proba"] = proba
    df["ensemble_pred_binary"] = (proba >= 0.5).astype(int)
    df["ensemble_confidence"] = np.abs(proba - 0.5) * 2.0
    df["model_version"] = MODEL_VERSION

    if "top_headlines" not in df.columns:
        df["top_headlines"] = "[]"

    return df[[c for c in OUTPUT_COLS if c in df.columns]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit learned ensemble meta-model and score all rows."
    )
    parser.add_argument(
        "--input", default=None,
        help=f"Path to eval_dataset.csv (default: {PROCESSED_DATA_DIR / 'eval_dataset.csv'})",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output CSV path (default: {PROCESSED_DATA_DIR / 'final_ensemble_predictions.csv'})",
    )
    parser.add_argument(
        "--meta-output", default=None,
        help=f"Output joblib for meta model (default: {MODELS_DIR / 'ensemble_meta.joblib'})",
    )
    parser.add_argument(
        "--db", default=str(DATABASE_PATH),
        help="SQLite DB path (for SPY 5d return + n_headlines augmentation).",
    )
    parser.add_argument(
        "--no-temperature-scale", action="store_true",
        help="Disable final temperature scaling step.",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else PROCESSED_DATA_DIR / "eval_dataset.csv"
    output_path = Path(args.output) if args.output else PROCESSED_DATA_DIR / "final_ensemble_predictions.csv"
    meta_path = Path(args.meta_output) if args.meta_output else MODELS_DIR / "ensemble_meta.joblib"
    db_path = Path(args.db)

    print("=" * 70)
    print("BUILD ENSEMBLE PREDICTIONS (HGB + adaptive calibration + temperature)")
    print("=" * 70)
    print(f"Input            : {input_path}")
    print(f"Output           : {output_path}")
    print(f"Meta             : {meta_path}")
    print(f"DB (SPY + news)  : {db_path}")
    print(f"Temperature scale: {'off' if args.no_temperature_scale else 'on'}")
    print()

    print("Step 1/4  Loading eval_dataset.csv ...")
    df = load_eval_dataset(input_path)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")
    print(f"  Rows with news: {int(df['has_news'].sum()):,}")

    print("Step 2/4  Augmenting features (SPY, n_headlines, confidences) ...")
    df = _augment(df, db_path)

    print("Step 3/4  Fitting meta-model ...")
    meta = fit_meta_model(df, temperature_scale=not args.no_temperature_scale)

    print("Step 4/4  Scoring + saving ...")
    out = compute_ensemble(df, meta)

    up = int((out["ensemble_pred_binary"] == 1).sum())
    down = int((out["ensemble_pred_binary"] == 0).sum())
    print(f"  Predictions    : {up} up / {down} down")
    print(f"  Avg confidence : {out['ensemble_confidence'].mean():.3f}")
    if meta.get("calibration_method"):
        print(f"  Calibration    : {meta['calibration_method']} (n_val={meta.get('n_val', '?')})")

    if "split" in out.columns and "actual_binary" in out.columns:
        for split in ["train", "val", "test"]:
            sub = out[(out["split"] == split) & out["actual_binary"].notna()]
            if len(sub) == 0:
                continue
            acc = (sub["ensemble_pred_binary"] == sub["actual_binary"]).mean()
            print(f"  {split:5s} accuracy ({len(sub):,} rows): {acc:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  Saved {len(out):,} rows to {output_path}")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "meta": meta["model"],
            "raw_model": meta["raw_model"],
            "temperature": meta["temperature"],
            "importances": meta["importances"],
            "features": META_FEATURES,
            "model_version": MODEL_VERSION,
        },
        meta_path,
    )
    print(f"  Saved meta-model to {meta_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
