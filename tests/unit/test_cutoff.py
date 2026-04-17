"""Pin the 4 PM ET news -> label_date mapping used by the NLP pipeline.

These tests exist to prevent the regression we fixed in `src/models/news_pipeline.py`:
the old `_published_at_to_prediction_date` in `scripts/train_nlp.py` always
shifted by +1 calendar day regardless of time-of-day, which misaligned every
headline with its label.
"""

import pytest

from src.models.news_pipeline import map_published_to_label_date


class TestCutoffMapping:
    def test_before_cutoff_monday(self):
        """Mon 3 PM ET news informs Mon->Tue direction -> label_date = Monday."""
        assert map_published_to_label_date("2026-03-16T15:30:00-04:00") == "2026-03-16"

    def test_after_cutoff_monday(self):
        """Mon 5 PM ET news informs Tue->Wed direction -> label_date = Tuesday."""
        assert map_published_to_label_date("2026-03-16T17:05:00-04:00") == "2026-03-17"

    def test_before_cutoff_friday(self):
        """Fri 3 PM ET -> label_date = Friday (predicts Fri->Mon)."""
        assert map_published_to_label_date("2026-03-20T15:00:00-04:00") == "2026-03-20"

    def test_after_cutoff_friday(self):
        """Fri 5 PM ET -> label_date = next trading day = Monday."""
        assert map_published_to_label_date("2026-03-20T17:00:00-04:00") == "2026-03-23"

    def test_weekend_news_bucketed_into_monday(self):
        """Saturday/Sunday news -> label_date = next Monday."""
        assert map_published_to_label_date("2026-03-21T10:00:00-04:00") == "2026-03-23"
        assert map_published_to_label_date("2026-03-22T22:00:00-04:00") == "2026-03-23"

    def test_accepts_z_suffix_utc(self):
        """Timestamps ending in 'Z' are treated as UTC and converted to ET."""
        # 2026-03-16T19:00:00Z = 2026-03-16T15:00:00-04:00 ET -> Monday label.
        assert map_published_to_label_date("2026-03-16T19:00:00Z") == "2026-03-16"

    def test_numeric_timestamp(self):
        """Finnhub-style unix seconds are accepted."""
        # 1773946800 = 2026-03-19 19:00 UTC = 15:00 ET (before cutoff) -> Thursday
        ts = "1773946800"
        assert map_published_to_label_date(ts) == "2026-03-19"

    def test_invalid_input_returns_none(self):
        assert map_published_to_label_date("") is None
        assert map_published_to_label_date("not-a-date") is None

    def test_consistency_before_vs_after_cutoff(self):
        """Same trading day: before-cutoff label < after-cutoff label."""
        before = map_published_to_label_date("2026-06-03T14:59:00-04:00")
        after = map_published_to_label_date("2026-06-03T16:01:00-04:00")
        assert before < after

    @pytest.mark.parametrize("utc_time,expected_label", [
        # UTC 14:00 -> ET 10:00 (before cutoff) -> Monday
        ("2026-04-06T14:00:00+00:00", "2026-04-06"),
        # UTC 20:00 -> ET 16:00 (AT cutoff -> after) -> Tuesday
        ("2026-04-06T20:00:00+00:00", "2026-04-07"),
        # UTC 23:59 -> ET 19:59 (after cutoff) -> Tuesday
        ("2026-04-06T23:59:00+00:00", "2026-04-07"),
    ])
    def test_utc_parametrized(self, utc_time, expected_label):
        assert map_published_to_label_date(utc_time) == expected_label
