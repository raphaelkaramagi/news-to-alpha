"""Validation-set threshold tuning and calibration quality guards."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Minimum spread required on validation probabilities before we trust calibration
# or a tuned threshold. Sigmoid prefit calibrators often collapse scores when val
# is small or the base model is already well-separated.
MIN_PROBA_STD = 0.05
MIN_CALIB_STD_RATIO = 0.30


def calibration_preserves_spread(
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    *,
    min_std: float = MIN_PROBA_STD,
    min_ratio: float = MIN_CALIB_STD_RATIO,
) -> bool:
    """Return True when calibration keeps enough probability spread."""
    raw = np.asarray(raw_proba, dtype=float)
    cal = np.asarray(cal_proba, dtype=float)
    if raw.size == 0 or cal.size == 0:
        return False
    raw_std = float(raw.std())
    cal_std = float(cal.std())
    if cal_std < min_std:
        return False
    if raw_std > 1e-6 and cal_std / raw_std < min_ratio:
        return False
    return True


def tune_threshold_balanced_accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick cutoff in [0.35, 0.65] maximizing balanced accuracy."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    if float(p.std()) < MIN_PROBA_STD:
        log.warning(
            "Skipping threshold tuning (proba std=%.4f < %.2f); using 0.5",
            float(p.std()), MIN_PROBA_STD,
        )
        return 0.5
    best_t, best_ba = 0.5, -1.0
    for t in np.linspace(0.35, 0.65, 61):
        pred = (p >= t).astype(int)
        up = int((y == 1).sum())
        down = int((y == 0).sum())
        r_up = int(((pred == 1) & (y == 1)).sum()) / max(up, 1)
        r_down = int(((pred == 0) & (y == 0)).sum()) / max(down, 1)
        ba = (r_up + r_down) / 2.0
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(proba, dtype=float) >= threshold).astype(int)
