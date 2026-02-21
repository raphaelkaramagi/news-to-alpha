"""Tests for chronological dataset splitting."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.data_processing.dataset_split import DatasetSplitter
from src.database.schema import DatabaseSchema


@pytest.fixture
def test_db(tmp_path):
    """Create a temp database with 10 days of fake price data for 2 tickers."""
    db_path = tmp_path / "test.db"
    DatabaseSchema(db_path).create_all_tables()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 10 consecutive trading dates (weekdays only)
    dates = [
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
        "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    ]

    for ticker in ["AAPL", "TSLA"]:
        for i, date in enumerate(dates):
            cursor.execute(
                "INSERT INTO prices (ticker, date, close, volume) VALUES (?, ?, ?, ?)",
                (ticker, date, 100.0 + i, 1000000),
            )

    conn.commit()
    conn.close()
    return db_path


class TestDatasetSplitter:
    def test_default_split_ratios(self, test_db, tmp_path):
        """70/15/15 split on 10 dates → 7 train, 1-2 val, 1-2 test."""
        splitter = DatasetSplitter(test_db)
        # Override the save path so it writes to tmp
        from unittest.mock import patch
        with patch("src.data_processing.dataset_split.PROCESSED_DATA_DIR", tmp_path):
            result = splitter.split()

        assert len(result["train"]["dates"]) == 7
        assert len(result["val"]["dates"]) >= 1
        assert len(result["test"]["dates"]) >= 1

        # All dates should be accounted for
        total = (
            len(result["train"]["dates"])
            + len(result["val"]["dates"])
            + len(result["test"]["dates"])
        )
        assert total == 10

    def test_no_date_overlap(self, test_db, tmp_path):
        """Train, val, and test dates must never overlap."""
        splitter = DatasetSplitter(test_db)
        from unittest.mock import patch
        with patch("src.data_processing.dataset_split.PROCESSED_DATA_DIR", tmp_path):
            result = splitter.split()

        train_set = set(result["train"]["dates"])
        val_set = set(result["val"]["dates"])
        test_set = set(result["test"]["dates"])

        assert train_set & val_set == set()
        assert train_set & test_set == set()
        assert val_set & test_set == set()

    def test_chronological_order(self, test_db, tmp_path):
        """All train dates must come before val dates, which come before test dates."""
        splitter = DatasetSplitter(test_db)
        from unittest.mock import patch
        with patch("src.data_processing.dataset_split.PROCESSED_DATA_DIR", tmp_path):
            result = splitter.split()

        # Last train date must be before first val date
        assert result["train"]["dates"][-1] < result["val"]["dates"][0]

        # Last val date must be before first test date
        assert result["val"]["dates"][-1] < result["test"]["dates"][0]

    def test_price_counts_match(self, test_db, tmp_path):
        """Total price rows across splits should equal total in DB."""
        splitter = DatasetSplitter(test_db)
        from unittest.mock import patch
        with patch("src.data_processing.dataset_split.PROCESSED_DATA_DIR", tmp_path):
            result = splitter.split()

        total_prices = sum(info["prices"] for info in result.values())
        assert total_prices == 20  # 10 dates * 2 tickers

    def test_split_info_saved(self, test_db, tmp_path):
        """split_info.json should be created in the processed directory."""
        splitter = DatasetSplitter(test_db)
        from unittest.mock import patch
        with patch("src.data_processing.dataset_split.PROCESSED_DATA_DIR", tmp_path):
            splitter.split()

        saved = tmp_path / "split_info.json"
        assert saved.exists()

        data = json.loads(saved.read_text())
        assert "splits" in data
        assert set(data["splits"].keys()) == {"train", "val", "test"}

    def test_too_few_dates_raises(self, tmp_path):
        """Splitting fewer than 3 dates should raise ValueError."""
        db_path = tmp_path / "tiny.db"
        DatabaseSchema(db_path).create_all_tables()

        conn = sqlite3.connect(db_path)
        conn.execute("INSERT INTO prices (ticker, date, close) VALUES ('AAPL', '2026-01-05', 100)")
        conn.execute("INSERT INTO prices (ticker, date, close) VALUES ('AAPL', '2026-01-06', 101)")
        conn.commit()
        conn.close()

        splitter = DatasetSplitter(db_path)
        with pytest.raises(ValueError, match="at least 3"):
            splitter.split()
