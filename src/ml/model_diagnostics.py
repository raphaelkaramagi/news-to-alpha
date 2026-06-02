"""Shared training / evaluation diagnostics for base models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def proba_stats(proba: np.ndarray) -> dict[str, float]:
    p = np.asarray(proba, dtype=float)
    if p.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "up_pct": float("nan")}
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(p)),
        "mean": float(p.mean()),
        "std": float(p.std()),
        "min": float(p.min()),
        "max": float(p.max()),
        "up_pct": float(pred.mean()),
    }


def split_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
    split_name: str = "split",
) -> dict[str, float | str]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)
    pred = (p >= threshold).astype(int)
    out: dict[str, float | str] = {
        "split": split_name,
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "up_pct_pred": float(pred.mean()),
    }
    stats = proba_stats(p)
    out.update({f"proba_{k}": v for k, v in stats.items() if k != "n"})
    if len(np.unique(y)) >= 2:
        try:
            out["auc"] = float(roc_auc_score(y, p))
        except ValueError:
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")
    return out


def print_split_metrics(metrics: dict[str, float | str], *, prefix: str = "") -> None:
    auc = metrics.get("auc", float("nan"))
    auc_s = f"{auc:.3f}" if isinstance(auc, float) and not np.isnan(auc) else "nan"
    print(
        f"{prefix}{metrics['split']:5s}  n={int(metrics['n']):5d}  "
        f"acc={metrics['accuracy']:.3f}  bal_acc={metrics['balanced_accuracy']:.3f}  "
        f"AUC={auc_s}  proba_std={metrics.get('proba_std', float('nan')):.4f}  "
        f"UP%={metrics.get('up_pct_pred', float('nan')):.1%}"
    )
    std = metrics.get("proba_std", 1.0)
    if isinstance(std, float) and std < 0.01:
        print(f"{prefix}  *** COLLAPSED: proba std < 0.01 ***")
    up = metrics.get("up_pct_pred", 0.5)
    if isinstance(up, float) and (up > 0.95 or up < 0.05):
        print(f"{prefix}  *** DEGENERATE: predictions almost all one class ***")


def per_ticker_auc(
    df: pd.DataFrame,
    *,
    y_col: str = "actual_binary",
    proba_col: str,
    ticker_col: str = "ticker",
    min_rows: int = 20,
) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker, grp in df.groupby(ticker_col):
        y = grp[y_col].dropna()
        if len(y) < min_rows or y.nunique() < 2:
            continue
        p = grp.loc[y.index, proba_col].astype(float)
        try:
            auc = float(roc_auc_score(y.astype(int), p))
        except ValueError:
            auc = float("nan")
        rows.append({"ticker": ticker, "n": int(len(y)), "auc": auc})
    if not rows:
        return pd.DataFrame(columns=["ticker", "n", "auc"])
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def print_per_ticker_auc(title: str, table: pd.DataFrame, *, top_n: int = 5) -> None:
    if table.empty:
        print(f"  {title}: (insufficient per-ticker data)")
        return
    print(f"  {title} (top {top_n} / bottom {top_n}):")
    for _, r in table.head(top_n).iterrows():
        print(f"    {r['ticker']:5s}  AUC={r['auc']:.3f}  n={int(r['n'])}")
    for _, r in table.tail(top_n).iterrows():
        print(f"    {r['ticker']:5s}  AUC={r['auc']:.3f}  n={int(r['n'])}")
