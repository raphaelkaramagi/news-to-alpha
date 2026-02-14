"""
Unit tests for the price data collector.

Run:  pytest tests/unit/test_price_collector.py -v
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.database.schema import DatabaseSchema
from src.data_collection.price_collector import PriceCollector


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    """Create a temporary SQLite database with the full schema."""
    db_path = str(tmp_path / "test.db")
    DatabaseSchema(db_path).create_all_tables()
    return db_path


class TestPriceCollector:
    """Tests for PriceCollector."""

    def test_collect_valid_ticker(self, tmp_db: str) -> None:
        """Collecting AAPL for a recent window should succeed."""
        collector = PriceCollector(db_path=tmp_db)
        stats = collector.collect(["AAPL"], "2026-01-20", "2026-02-07")

        assert "AAPL" in stats["tickers_succeeded"]
        assert stats["rows_added"] > 0
        assert len(stats["tickers_failed"]) == 0

    def test_collect_invalid_ticker(self, tmp_db: str) -> None:
        """An invalid ticker should be recorded as failed, not crash."""
        collector = PriceCollector(db_path=tmp_db)
        stats = collector.collect(["ZZZZZINVALID"], "2026-01-20", "2026-02-07")

        # yfinance returns empty for fake tickers — recorded as failure
        assert stats["rows_added"] == 0

    def test_no_duplicate_inserts(self, tmp_db: str) -> None:
        """Running collection twice should skip duplicates."""
        collector = PriceCollector(db_path=tmp_db)
        args = (["AAPL"], "2026-01-27", "2026-02-07")

        first = collector.collect(*args)
        second = collector.collect(*args)

        assert first["rows_added"] > 0
        assert second["rows_added"] == 0
        assert second["duplicates_skipped"] == first["rows_added"]

    def test_run_logged(self, tmp_db: str) -> None:
        """Each collection run should write to run_log."""
        collector = PriceCollector(db_path=tmp_db)
        collector.collect(["AAPL"], "2026-01-27", "2026-02-07")

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute(
            "SELECT COUNT(*) FROM run_log WHERE run_type='price_collection'"
        ).fetchone()[0]
        conn.close()

        assert rows >= 1
