"""Compute incremental collection date ranges from SQLite coverage."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from src.config import DATABASE_PATH, MARKET_INDEX_TICKER, TICKERS


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def latest_price_date(
    tickers: list[str],
    db_path: str | Path = DATABASE_PATH,
) -> date | None:
    """Latest price bar shared across all requested tickers (weakest link)."""
    if not tickers or not Path(db_path).exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        placeholders = ",".join("?" * len(tickers))
        row = conn.execute(
            f"""
            SELECT MIN(latest) FROM (
                SELECT MAX(date) AS latest
                FROM prices
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            )
            """,
            tickers,
        ).fetchone()
        return _parse_day(row[0] if row else None)
    finally:
        conn.close()


def latest_news_date(
    tickers: list[str],
    db_path: str | Path = DATABASE_PATH,
) -> date | None:
    """Earliest per-ticker max news date across the universe (weakest link)."""
    if not tickers or not Path(db_path).exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        has_news = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='news'"
        ).fetchone()
        if not has_news:
            return None
        placeholders = ",".join("?" * len(tickers))
        row = conn.execute(
            f"""
            SELECT MIN(latest) FROM (
                SELECT MAX(substr(published_at, 1, 10)) AS latest
                FROM news
                WHERE ticker IN ({placeholders})
                GROUP BY ticker
            )
            """,
            tickers,
        ).fetchone()
        return _parse_day(row[0] if row else None)
    finally:
        conn.close()


def compute_collection_window(
    tickers: list[str] | None = None,
    *,
    buffer_days: int = 5,
    min_days: int = 7,
    max_days: int = 60,
    db_path: str | Path = DATABASE_PATH,
    end: date | None = None,
) -> tuple[str, str, dict]:
    """Return (start_date, end_date, info) for incremental daily collection.

    Uses the **minimum** of per-ticker latest dates so a single lagging symbol
    does not leave gaps. Applies ``buffer_days`` overlap for late revisions,
    clamps to ``[min_days, max_days]`` lookback from *end*.
    """
    tickers = tickers or list(TICKERS)
    end_d = end or date.today()
    end_s = end_d.isoformat()

    price_latest = latest_price_date(tickers, db_path)
    news_latest = latest_news_date(tickers, db_path)

    if price_latest is None:
        start_d = end_d - timedelta(days=max_days)
        mode = "empty_db"
        gap_days = max_days
    else:
        gap_days = max(0, (end_d - price_latest).days)
        lookback = max(min_days, min(max_days, gap_days + buffer_days))
        start_d = end_d - timedelta(days=lookback)
        mode = "incremental"

    info = {
        "mode": mode,
        "end_date": end_s,
        "start_date": start_d.isoformat(),
        "lookback_calendar_days": (end_d - start_d).days,
        "gap_since_price_latest": gap_days if price_latest else None,
        "price_latest": price_latest.isoformat() if price_latest else None,
        "news_latest": news_latest.isoformat() if news_latest else None,
        "buffer_days": buffer_days,
        "min_days": min_days,
        "max_days": max_days,
    }
    return start_d.isoformat(), end_s, info


def universe_tickers(configured: list[str] | None = None) -> list[str]:
    """Project tickers plus regime symbols used by the LSTM pipeline."""
    base = list(configured or TICKERS)
    for sym in (MARKET_INDEX_TICKER, "VIX"):
        if sym not in base:
            base.append(sym)
    return base
