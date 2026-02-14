"""
Unit tests for data standardization utilities.

Run:  pytest tests/unit/test_standardization.py -v
"""

from src.data_processing.standardization import DataStandardizer


class TestDataStandardizer:
    def test_standardize_date_string(self) -> None:
        assert DataStandardizer.standardize_date("February 7, 2026") == "2026-02-07"

    def test_standardize_date_iso(self) -> None:
        assert DataStandardizer.standardize_date("2026-02-07") == "2026-02-07"

    def test_standardize_timestamp_unix(self) -> None:
        result = DataStandardizer.standardize_timestamp(1707321180)
        assert result.startswith("2024-02-07")
        assert "-05:00" in result or "-04:00" in result  # ET offset

    def test_cutoff_before_4pm(self) -> None:
        """Article at 2 PM ET on Monday -> predicts Tuesday."""
        result = DataStandardizer.apply_cutoff_rule(
            "2026-02-09T14:00:00-05:00"
        )
        assert result == "2026-02-10"

    def test_cutoff_after_4pm(self) -> None:
        """Article at 5 PM ET on Monday -> predicts Wednesday."""
        result = DataStandardizer.apply_cutoff_rule(
            "2026-02-09T17:00:00-05:00"
        )
        assert result == "2026-02-11"
