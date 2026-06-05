"""Guards against LSTM probability collapse (identical day-to-day P(up))."""

import numpy as np

from scripts.train_lstm import _is_degenerate_calibration


def test_collapsed_output_is_flagged():
    # A few plateaus across many rows -> degenerate (the bug we fixed).
    collapsed = np.array([0.5457] * 460 + [0.5128] * 200 + [0.5511] * 150)
    assert _is_degenerate_calibration(collapsed)


def test_near_constant_output_is_flagged():
    near_const = np.full(500, 0.53) + np.random.normal(0, 0.001, 500)
    assert _is_degenerate_calibration(near_const)


def test_healthy_spread_passes():
    rng = np.random.default_rng(0)
    healthy = np.clip(rng.normal(0.5, 0.12, 1000), 0.02, 0.98)
    assert not _is_degenerate_calibration(healthy)
