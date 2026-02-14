"""
Unit tests for database schema creation.

Run:  pytest tests/unit/test_schema.py -v
"""

import sqlite3
from pathlib import Path

import pytest

from src.database.schema import DatabaseSchema


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    DatabaseSchema(db_path).create_all_tables()
    return db_path


EXPECTED_TABLES = {"prices", "news", "labels", "predictions", "run_log"}


class TestDatabaseSchema:
    def test_all_tables_created(self, tmp_db: str) -> None:
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert EXPECTED_TABLES.issubset(tables)

    def test_idempotent(self, tmp_db: str) -> None:
        """Running create_all_tables twice should not raise."""
        DatabaseSchema(tmp_db).create_all_tables()

    def test_prices_unique_constraint(self, tmp_db: str) -> None:
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO prices (ticker, date, close) VALUES ('AAPL', '2026-02-01', 100.0)"
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO prices (ticker, date, close) VALUES ('AAPL', '2026-02-01', 101.0)"
            )
        conn.close()

    def test_news_url_unique(self, tmp_db: str) -> None:
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO news (url, ticker, title, published_at) "
            "VALUES ('https://example.com/1', 'AAPL', 'Test', '2026-02-01T10:00:00-05:00')"
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO news (url, ticker, title, published_at) "
                "VALUES ('https://example.com/1', 'AAPL', 'Dupe', '2026-02-01T11:00:00-05:00')"
            )
        conn.close()
