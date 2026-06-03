"""Tests for score_models outcome backfill helpers."""

from scripts.score_models import _abs_return_pct_from_label


def test_abs_return_pct_from_label_is_already_percent_points():
    # label_generator stores pct_return_1d = ((close_t1 - close_t) / close_t) * 100
    assert _abs_return_pct_from_label(-1.8426) == 1.8426
    assert _abs_return_pct_from_label(0.5) == 0.5
