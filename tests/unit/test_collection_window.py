"""Tests for incremental collection window helpers."""

from datetime import date, timedelta

import pytest

from src.utils.collection_window import compute_collection_window, latest_price_date


@pytest.fixture
def db_with_prices(tmp_path):
    import sqlite3

    db = tmp_path / "test.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE prices (ticker TEXT, date TEXT, close REAL)"
    )
    conn.execute(
        "INSERT INTO prices VALUES ('AAPL', ?, 100)",
        ((date.today() - timedelta(days=2)).isoformat(),),
    )
    conn.execute(
        "INSERT INTO prices VALUES ('NVDA', ?, 200)",
        ((date.today() - timedelta(days=2)).isoformat(),),
    )
    conn.commit()
    conn.close()
    return db


def test_latest_price_date(db_with_prices):
    latest = latest_price_date(["AAPL", "NVDA"], db_with_prices)
    assert latest == date.today() - timedelta(days=2)


def test_incremental_window_uses_min_not_max(db_with_prices):
    start, end, info = compute_collection_window(
        ["AAPL", "NVDA"],
        buffer_days=2,
        min_days=7,
        max_days=60,
        db_path=db_with_prices,
        end=date.today(),
    )
    assert info["mode"] == "incremental"
    assert info["gap_since_price_latest"] == 2
    # gap 2 + buffer 2 = 4, clamped up to min_days 7
    assert info["lookback_calendar_days"] == 7
    assert start <= end


def test_empty_db_uses_max_lookback(tmp_path):
    import sqlite3

    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE prices (ticker TEXT, date TEXT, close REAL)")
    conn.commit()
    conn.close()

    start, end, info = compute_collection_window(
        ["AAPL"],
        max_days=30,
        db_path=db,
        end=date(2026, 5, 28),
    )
    assert info["mode"] == "empty_db"
    assert info["lookback_calendar_days"] == 30
    assert start == "2026-04-28"
