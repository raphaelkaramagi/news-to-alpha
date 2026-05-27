"""NYSE trading-calendar utilities.

Used by the Flask API to report accurate freshness vs naive weekday counting.
Falls back to a simple Mon–Fri rule when exchange_calendars is unavailable so
the server still starts even on slim production images that don't include it.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional


def _get_nyse():
    try:
        import exchange_calendars as xcals
        return xcals.get_calendar("XNYS")
    except Exception:
        return None


def last_trading_session(as_of: Optional[date] = None) -> date:
    """Return the most recent completed NYSE trading session."""
    today = as_of or date.today()
    cal = _get_nyse()
    if cal is not None:
        import pandas as pd
        sessions = cal.sessions_in_range(
            pd.Timestamp(today - timedelta(days=14)),
            pd.Timestamp(today),
        )
        past = [s.date() for s in sessions if s.date() < today]
        if past:
            return past[-1]
        return today - timedelta(days=1)

    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def sessions_between(start: date, end: date) -> int:
    """Count NYSE trading sessions in the range [start, end) exclusive of end."""
    if end <= start:
        return 0
    cal = _get_nyse()
    if cal is not None:
        import pandas as pd
        sessions = cal.sessions_in_range(
            pd.Timestamp(start),
            pd.Timestamp(end - timedelta(days=1)),
        )
        return len(sessions)

    count = 0
    d = start
    while d < end:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return count


def prediction_lag_sessions(
    latest_pred_date_str: Optional[str],
    latest_price_date_str: Optional[str],
) -> int:
    """Trading sessions predictions lag behind the latest price date."""
    if not latest_pred_date_str or not latest_price_date_str:
        return -1
    try:
        latest_pred = datetime.strptime(latest_pred_date_str[:10], "%Y-%m-%d").date()
        latest_price = datetime.strptime(latest_price_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return -1
    if latest_pred >= latest_price:
        return 0
    return sessions_between(latest_pred + timedelta(days=1), latest_price + timedelta(days=1))


def sessions_behind(latest_pred_date_str: Optional[str], horizon: int = 1) -> int:
    """Legacy calendar-based lag (kept for tests). Prefer prediction_lag_sessions."""
    if not latest_pred_date_str:
        return -1
    try:
        latest = datetime.strptime(latest_pred_date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return -1

    last_session = last_trading_session()
    expected = last_session
    cal = _get_nyse()
    if cal is not None:
        import pandas as pd
        sessions = cal.sessions_in_range(
            pd.Timestamp(last_session - timedelta(days=30)),
            pd.Timestamp(last_session),
        )
        session_dates = [s.date() for s in sessions]
        if len(session_dates) > horizon:
            expected = session_dates[-1 - horizon]
        else:
            expected = session_dates[0] if session_dates else last_session
    else:
        for _ in range(horizon):
            expected -= timedelta(days=1)
            while expected.weekday() >= 5:
                expected -= timedelta(days=1)

    if latest >= expected:
        return 0
    return sessions_between(latest + timedelta(days=1), expected + timedelta(days=1))
