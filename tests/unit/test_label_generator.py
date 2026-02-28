"""Tests for label generation."""

import sqlite3
import pytest

from src.data_processing.label_generator import LabelGenerator
from src.database.schema import DatabaseSchema


@pytest.fixture
def test_db(tmp_path):
    """Create a temp DB with 5 days of price data for AAPL."""
    db_path = tmp_path / "test.db"
    DatabaseSchema(db_path).create_all_tables()

    conn = sqlite3.connect(db_path)
    # 5 consecutive trading days — prices go: 100, 102, 101, 105, 103
    # So labels should be: up, down, up, down (4 labels, last day has none)
    prices = [
        ("AAPL", "2026-01-05", 100.0),
        ("AAPL", "2026-01-06", 102.0),  # up from 100
        ("AAPL", "2026-01-07", 101.0),  # down from 102
        ("AAPL", "2026-01-08", 105.0),  # up from 101
        ("AAPL", "2026-01-09", 103.0),  # down from 105
    ]
    for ticker, date, close in prices:
        conn.execute(
            "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)",
            (ticker, date, close),
        )
    conn.commit()
    conn.close()
    return db_path


class TestLabelGenerator:
    def test_correct_label_count(self, test_db):
        """5 price rows → 4 labels (last day has no next-day data)."""
        gen = LabelGenerator(test_db)
        result = gen.generate(["AAPL"])
        assert result["tickers"]["AAPL"]["labels"] == 4

    def test_up_down_values(self, test_db):
        """Check the actual up/down labels match expected pattern."""
        gen = LabelGenerator(test_db)
        gen.generate(["AAPL"])

        conn = sqlite3.connect(test_db)
        labels = conn.execute(
            "SELECT date, label_binary FROM labels WHERE ticker='AAPL' ORDER BY date"
        ).fetchall()
        conn.close()

        # 100→102 = up, 102→101 = down, 101→105 = up, 105→103 = down
        assert labels == [
            ("2026-01-05", 1),
            ("2026-01-06", 0),
            ("2026-01-07", 1),
            ("2026-01-08", 0),
        ]

    def test_pct_return_calculated(self, test_db):
        """Percentage return should be stored correctly."""
        gen = LabelGenerator(test_db)
        gen.generate(["AAPL"])

        conn = sqlite3.connect(test_db)
        row = conn.execute(
            "SELECT label_return FROM labels WHERE ticker='AAPL' AND date='2026-01-05'"
        ).fetchone()
        conn.close()

        # 100 → 102 = +2%
        assert abs(row[0] - 2.0) < 0.01

    def test_no_duplicates_on_rerun(self, test_db):
        """Running generate twice should skip existing labels."""
        gen = LabelGenerator(test_db)
        first = gen.generate(["AAPL"])
        second = gen.generate(["AAPL"])

        assert first["tickers"]["AAPL"]["labels"] == 4
        assert second["tickers"]["AAPL"]["labels"] == 0
        assert second["tickers"]["AAPL"]["skipped_dupes"] == 4

    def test_skips_ticker_with_one_row(self, tmp_path):
        """A ticker with only 1 price row can't produce labels."""
        db_path = tmp_path / "tiny.db"
        DatabaseSchema(db_path).create_all_tables()

        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO prices (ticker, date, close) VALUES ('X', '2026-01-05', 50)")
        conn.commit()
        conn.close()

        gen = LabelGenerator(db_path)
        result = gen.generate(["X"])
        assert result["tickers"]["X"]["labels"] == 0
