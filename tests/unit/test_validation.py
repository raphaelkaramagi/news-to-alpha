"""Tests for price and news data validation."""

import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.database.schema import DatabaseSchema
from src.data_processing.price_validation import PriceDataValidator
from src.data_processing.news_validation import NewsDataValidator


@pytest.fixture
def price_db(tmp_path):
    """Database with clean and problematic price data."""
    db_path = str(tmp_path / "validation.db")
    DatabaseSchema(db_path).create_all_tables()

    conn = sqlite3.connect(db_path)

    # 10 normal rows for AAPL
    dates = pd.bdate_range("2026-01-05", periods=10)
    for i, date in enumerate(dates):
        conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("AAPL", date.strftime("%Y-%m-%d"),
             150 + i, 152 + i, 148 + i, 151 + i, 5000000),
        )

    # Row with NULL open (missing data)
    conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
        "VALUES ('BAD', '2026-01-05', NULL, 100, 99, 100, 1000)"
    )

    # Zero-volume row
    conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
        "VALUES ('BAD', '2026-01-06', 100, 101, 99, 100, 0)"
    )

    # Large price jump (> 20%)
    conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
        "VALUES ('BAD', '2026-01-07', 100, 130, 99, 125, 1000)"
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def news_db(tmp_path):
    """Database with clean and problematic news data."""
    db_path = str(tmp_path / "news_val.db")
    DatabaseSchema(db_path).create_all_tables()

    conn = sqlite3.connect(db_path)

    # Normal articles
    conn.execute(
        "INSERT INTO news (ticker, title, url, source, published_at) "
        "VALUES ('AAPL', 'Apple rises', 'https://a.com/1', 'reuters', "
        "'2026-01-05T10:00:00-05:00')"
    )
    conn.execute(
        "INSERT INTO news (ticker, title, url, source, published_at) "
        "VALUES ('AAPL', 'Apple new product', 'https://a.com/2', 'bloomberg', "
        "'2026-01-06T14:00:00-05:00')"
    )

    # Article with missing title
    conn.execute(
        "INSERT INTO news (ticker, title, url, source, published_at) "
        "VALUES ('TSLA', '', 'https://a.com/3', 'unknown', '2026-01-05T10:00:00-05:00')"
    )

    # Article with missing source
    conn.execute(
        "INSERT INTO news (ticker, title, url, source, published_at) "
        "VALUES ('TSLA', 'Tesla update', 'https://a.com/4', '', '2026-01-05T11:00:00-05:00')"
    )

    conn.commit()
    conn.close()
    return db_path


class TestPriceDataValidator:
    def test_detects_missing_values(self, price_db):
        """Rows with NULL OHLCV should be flagged."""
        validator = PriceDataValidator(db_path=price_db)
        results = validator.validate(["AAPL", "BAD"])

        missing = results["missing_data"]
        bad_tickers = [r["ticker"] for r in missing]
        assert "BAD" in bad_tickers
        assert "AAPL" not in bad_tickers

    def test_detects_zero_volume(self, price_db):
        """Zero-volume rows should be flagged."""
        validator = PriceDataValidator(db_path=price_db)
        results = validator.validate(["BAD"])
        assert len(results["volume_anomalies"]) >= 1

    def test_detects_price_jumps(self, price_db):
        """Consecutive day jumps >20% should be flagged."""
        validator = PriceDataValidator(db_path=price_db)
        results = validator.validate(["BAD"])
        assert len(results["price_anomalies"]) >= 1

    def test_coverage_has_correct_counts(self, price_db):
        """Coverage should report correct number of days per ticker."""
        validator = PriceDataValidator(db_path=price_db)
        results = validator.validate(["AAPL"])
        coverage = results["coverage"]
        assert len(coverage) == 1
        assert coverage[0]["ticker"] == "AAPL"
        assert coverage[0]["days_collected"] == 10

    def test_clean_data_no_issues(self, price_db):
        """AAPL-only validation should show no anomalies."""
        validator = PriceDataValidator(db_path=price_db)
        results = validator.validate(["AAPL"])
        assert len(results["missing_data"]) == 0
        assert len(results["volume_anomalies"]) == 0


class TestNewsDataValidator:
    def test_detects_missing_fields(self, news_db):
        """Articles with empty title or source should be flagged."""
        validator = NewsDataValidator(db_path=news_db)
        results = validator.validate(["AAPL", "TSLA"])
        assert len(results["missing_fields"]) > 0

    def test_no_future_timestamps(self, news_db):
        """Articles with past timestamps should not be flagged."""
        validator = NewsDataValidator(db_path=news_db)
        results = validator.validate(["AAPL", "TSLA"])
        assert len(results["future_timestamps"]) == 0

    def test_no_duplicates(self, news_db):
        """Unique (url, ticker) pairs should show no duplicates."""
        validator = NewsDataValidator(db_path=news_db)
        results = validator.validate(["AAPL", "TSLA"])
        assert len(results["duplicate_url_ticker_pairs"]) == 0

    def test_distribution_counts(self, news_db):
        """Article counts per ticker should be accurate."""
        validator = NewsDataValidator(db_path=news_db)
        results = validator.validate(["AAPL", "TSLA"])
        dist = {r["ticker"]: r["article_count"] for r in results["articles_per_ticker"]}
        assert dist.get("AAPL") == 2
        assert dist.get("TSLA") == 2

    def test_empty_db_no_crash(self, tmp_path):
        """Validating an empty database should return empty results, not crash."""
        db_path = str(tmp_path / "empty.db")
        DatabaseSchema(db_path).create_all_tables()

        validator = NewsDataValidator(db_path=db_path)
        results = validator.validate(["AAPL"])
        assert len(results["missing_fields"]) == 0
        assert len(results["future_timestamps"]) == 0
        assert len(results["duplicate_url_ticker_pairs"]) == 0
