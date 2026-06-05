#!/usr/bin/env python3
"""Walk-forward (purged, embargoed) evaluation harness.

Judges predictive skill across *many* temporal folds instead of one 15% slice,
so a model has to be right repeatedly over long durations to look good. Also the
research vehicle for alternative targets (next-day vs weekly vs cross-sectional
vs large-move) — see ``--target``.

Design
------
- Features: technical indicators per (ticker, date) from the prices DB
  (scale-invariant subset by default; add levels with ``--include-levels``).
- Targets (``--target``):
    next_day        : sign of close(t+1)/close(t) - 1            (the product default)
    weekly          : sign of close(t+5)/close(t) - 1            (5-session horizon)
    cross_sectional : does this ticker beat the cross-sectional MEDIAN next-day
                      return? (market-neutral ranking — usually most learnable)
    large_move      : next_day direction, evaluated ONLY on days whose realized
                      |return| >= --move-threshold (does direction work on big days?)
- Splits: expanding-window walk-forward. Train = all dates up to a cutoff; an
  ``--embargo`` gap of trading days is purged; test = the next ``--test-size``
  dates. Repeats for ``--folds`` folds. The embargo prevents the label horizon
  (and any rolling feature) from straddling the train/test boundary.
- Model (``--model``): hgb (HistGradientBoosting) or logistic. Both calibrated
  per fold on an inner validation tail of the train window.

Reports per-fold and aggregate AUC, accuracy, balanced accuracy, Brier,
calibration error (reliability), and prediction-variance (to catch collapse).

Usage
-----
  python scripts/walk_forward_eval.py --target next_day
  python scripts/walk_forward_eval.py --target cross_sectional --model hgb
  python scripts/walk_forward_eval.py --target weekly --folds 8 --test-size 20
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, TICKERS  # noqa: E402
from src.features.sequence_generator import (  # noqa: E402
    LEVEL_FEATURES,
    SCALE_INVARIANT_FEATURES,
)
from src.features.technical_indicators import TechnicalIndicators  # noqa: E402

TARGETS = ("next_day", "weekly", "cross_sectional", "large_move")


# ---------------------------------------------------------------------------
# Feature + target construction
# ---------------------------------------------------------------------------

def build_panel(
    tickers: list[str],
    *,
    include_levels: bool = False,
    include_fundamentals: bool = False,
    weekly_horizon: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """One row per (ticker, date) with features and all candidate targets."""
    feature_cols = list(SCALE_INVARIANT_FEATURES)
    if include_levels:
        feature_cols = feature_cols + list(LEVEL_FEATURES)

    ti = TechnicalIndicators()
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = ti.compute(ticker)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"date": "prediction_date"})
        df["prediction_date"] = pd.to_datetime(df["prediction_date"]).dt.strftime("%Y-%m-%d")
        df["ticker"] = ticker

        close = df["close"].astype(float)
        fwd1 = close.shift(-1) / close - 1.0
        fwdw = close.shift(-weekly_horizon) / close - 1.0
        df["ret_fwd_1"] = fwd1 * 100.0
        df["ret_fwd_w"] = fwdw * 100.0
        df["abs_ret_fwd_1"] = df["ret_fwd_1"].abs()
        df["y_next_day"] = (fwd1 > 0).astype("Int64")
        df["y_weekly"] = (fwdw > 0).astype("Int64")
        frames.append(df)

    if not frames:
        return pd.DataFrame(), feature_cols

    panel = pd.concat(frames, ignore_index=True)

    # Cross-sectional target: beat the same-day median next-day return.
    med = panel.groupby("prediction_date")["ret_fwd_1"].transform("median")
    panel["y_cross_sectional"] = (panel["ret_fwd_1"] > med).astype("Int64")

    if include_fundamentals:
        from src.features.fundamentals_features import add_earnings_proximity
        panel = add_earnings_proximity(panel)
        feature_cols = feature_cols + ["days_to_earnings", "earnings_window"]

    panel = panel.dropna(subset=feature_cols)
    return panel, feature_cols


def target_frame(panel: pd.DataFrame, target: str, move_threshold: float) -> pd.DataFrame:
    """Attach a unified ``y`` column and apply any row filter for the target."""
    df = panel.copy()
    if target == "next_day":
        df["y"] = df["y_next_day"]
    elif target == "weekly":
        df["y"] = df["y_weekly"]
    elif target == "cross_sectional":
        df["y"] = df["y_cross_sectional"]
    elif target == "large_move":
        df["y"] = df["y_next_day"]
        df = df[df["abs_ret_fwd_1"] >= move_threshold]
    else:
        raise ValueError(f"Unknown target {target}")
    return df.dropna(subset=["y"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Walk-forward splitter
# ---------------------------------------------------------------------------

@dataclass
class FoldSpec:
    fold: int
    train_dates: list[str]
    test_dates: list[str]


def make_folds(
    dates: list[str],
    *,
    folds: int,
    test_size: int,
    embargo: int,
    min_train: int,
) -> list[FoldSpec]:
    """Expanding-window folds anchored at the END of the calendar.

    The last ``folds * test_size`` dates form the test blocks; each fold trains
    on everything before its test block minus an ``embargo`` gap.
    """
    uniq = sorted(set(dates))
    out: list[FoldSpec] = []
    total_test = folds * test_size
    if len(uniq) < min_train + embargo + total_test:
        # Shrink folds to whatever fits.
        usable = max(0, len(uniq) - min_train - embargo)
        folds = max(1, usable // max(1, test_size))
        total_test = folds * test_size

    # walk forward from the end — most recent folds matter most for "is it still working?"
    first_test_idx = len(uniq) - total_test
    for k in range(folds):
        test_start = first_test_idx + k * test_size
        test_end = test_start + test_size
        test_block = uniq[test_start:test_end]
        # embargo gap: stop 60-day lstm window + label horizon leaking across the cut
        train_block = uniq[: max(0, test_start - embargo)]
        if len(train_block) < min_train or not test_block:
            continue
        out.append(FoldSpec(fold=k + 1, train_dates=train_block, test_dates=test_block))
    return out


# ---------------------------------------------------------------------------
# Model + metrics
# ---------------------------------------------------------------------------

def _make_model(name: str):
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            l2_regularization=1.0, random_state=42,
        )
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
        )
    raise ValueError(f"Unknown model {name}")


def _calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _fold_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score,
    )
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "ece": _calibration_error(y, p),
        "proba_std": float(np.std(p)),
        "proba_nunique": int(np.unique(np.round(p, 4)).size),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    target: str
    model: str
    per_fold: pd.DataFrame
    aggregate: dict = field(default_factory=dict)


def run_walk_forward(
    target: str,
    *,
    model: str = "hgb",
    folds: int = 6,
    test_size: int = 20,
    embargo: int = 6,
    min_train: int = 252,
    include_levels: bool = False,
    include_fundamentals: bool = False,
    move_threshold: float = 1.5,
    tickers: list[str] | None = None,
    val_tail_frac: float = 0.15,
) -> WalkForwardResult:
    tickers = tickers or list(TICKERS)
    panel, feature_cols = build_panel(
        tickers, include_levels=include_levels, include_fundamentals=include_fundamentals,
    )
    if panel.empty:
        raise RuntimeError("No feature rows built — is the prices DB populated?")
    df = target_frame(panel, target, move_threshold)

    fold_specs = make_folds(
        sorted(df["prediction_date"].unique()),
        folds=folds, test_size=test_size, embargo=embargo, min_train=min_train,
    )
    if not fold_specs:
        raise RuntimeError("Not enough history for the requested fold layout.")

    rows = []
    for spec in fold_specs:
        train = df[df["prediction_date"].isin(spec.train_dates)]
        test = df[df["prediction_date"].isin(spec.test_dates)]
        if train["y"].nunique() < 2 or test.empty:
            continue

        # Inner validation tail (chronological) for calibration.
        tr_dates = sorted(train["prediction_date"].unique())
        cut = int(len(tr_dates) * (1.0 - val_tail_frac))
        fit_dates = set(tr_dates[:cut]) or set(tr_dates)
        cal_dates = set(tr_dates[cut:])
        fit = train[train["prediction_date"].isin(fit_dates)]
        cal = train[train["prediction_date"].isin(cal_dates)]

        clf = _make_model(model)
        clf.fit(fit[feature_cols].to_numpy(float), fit["y"].astype(int).to_numpy())

        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        raw_test = clf.predict_proba(test[feature_cols].to_numpy(float))[:, 1]
        # Calibrate on the val tail when it has both classes; else use raw.
        if not cal.empty and cal["y"].nunique() == 2:
            raw_cal = clf.predict_proba(cal[feature_cols].to_numpy(float))[:, 1]
            # Platt scaling (robust on small folds; isotonic collapses).
            platt = LogisticRegression(C=1.0, max_iter=500)
            platt.fit(
                np.clip(raw_cal, 1e-6, 1 - 1e-6).reshape(-1, 1),
                cal["y"].astype(int).to_numpy(),
            )
            p_test = platt.predict_proba(raw_test.reshape(-1, 1))[:, 1]
        else:
            p_test = raw_test

        m = _fold_metrics(test["y"].astype(int).to_numpy(), p_test)
        m["fold"] = spec.fold
        m["train_days"] = len(spec.train_dates)
        m["test_start"] = spec.test_dates[0]
        m["test_end"] = spec.test_dates[-1]
        rows.append(m)

    per_fold = pd.DataFrame(rows)
    agg: dict = {}
    if not per_fold.empty:
        for col in ("auc", "accuracy", "balanced_acc", "brier", "ece", "proba_std"):
            agg[f"{col}_mean"] = float(per_fold[col].mean())
            agg[f"{col}_std"] = float(per_fold[col].std())
        agg["n_folds"] = int(len(per_fold))
        agg["n_test_total"] = int(per_fold["n"].sum())
    return WalkForwardResult(target=target, model=model, per_fold=per_fold, aggregate=agg)


def _print_result(res: WalkForwardResult) -> None:
    print("=" * 72)
    print(f"WALK-FORWARD EVAL  target={res.target}  model={res.model}")
    print("=" * 72)
    if res.per_fold.empty:
        print("No usable folds.")
        return
    cols = ["fold", "test_start", "test_end", "n", "auc", "accuracy",
            "balanced_acc", "brier", "ece", "proba_std", "proba_nunique"]
    show = res.per_fold[cols].copy()
    for c in ("auc", "accuracy", "balanced_acc", "brier", "ece", "proba_std"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    a = res.aggregate
    print("-" * 72)
    print(f"AUC      {a['auc_mean']:.4f} +/- {a['auc_std']:.4f}")
    print(f"Accuracy {a['accuracy_mean']:.4f} +/- {a['accuracy_std']:.4f}")
    print(f"BalAcc   {a['balanced_acc_mean']:.4f} +/- {a['balanced_acc_std']:.4f}")
    print(f"Brier    {a['brier_mean']:.4f}   ECE {a['ece_mean']:.4f}")
    print(f"folds={a['n_folds']}  test_rows={a['n_test_total']}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Purged walk-forward evaluation")
    ap.add_argument("--target", default="next_day", choices=TARGETS)
    ap.add_argument("--model", default="hgb", choices=["hgb", "logistic"])
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--test-size", type=int, default=20, help="Trading days per test fold")
    ap.add_argument("--embargo", type=int, default=6, help="Purged trading days between train/test")
    ap.add_argument("--min-train", type=int, default=252)
    ap.add_argument("--include-levels", action="store_true")
    ap.add_argument("--include-fundamentals", action="store_true",
                    help="Add earnings-proximity features (needs collect_fundamentals.py)")
    ap.add_argument("--move-threshold", type=float, default=1.5,
                    help="abs return %% cutoff for target=large_move")
    ap.add_argument("--all-targets", action="store_true",
                    help="Run every target and print a comparison table")
    ap.add_argument("--out", default=None, help="Optional CSV path for per-fold rows")
    args = ap.parse_args()

    targets = list(TARGETS) if args.all_targets else [args.target]
    summary_rows = []
    for tgt in targets:
        res = run_walk_forward(
            tgt, model=args.model, folds=args.folds, test_size=args.test_size,
            embargo=args.embargo, min_train=args.min_train,
            include_levels=args.include_levels,
            include_fundamentals=args.include_fundamentals,
            move_threshold=args.move_threshold,
        )
        _print_result(res)
        if not res.per_fold.empty:
            a = res.aggregate
            summary_rows.append({
                "target": tgt, "model": args.model,
                "auc": round(a["auc_mean"], 4), "auc_std": round(a["auc_std"], 4),
                "accuracy": round(a["accuracy_mean"], 4),
                "balanced_acc": round(a["balanced_acc_mean"], 4),
                "ece": round(a["ece_mean"], 4),
                "folds": a["n_folds"], "test_rows": a["n_test_total"],
            })
        out_path = args.out or (PROCESSED_DATA_DIR / f"walk_forward_{tgt}.csv")
        if not res.per_fold.empty:
            res.per_fold.to_csv(out_path, index=False)

    if len(summary_rows) > 1:
        print("\nTARGET COMPARISON")
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
