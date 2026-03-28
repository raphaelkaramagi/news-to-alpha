"""Date/time standardization and cutoff-time logic."""

from datetime import datetime, date as date_type, timedelta
import pandas as pd
import pytz

# US market holidays (month, day) that fall on fixed dates.
# Floating holidays (e.g. Thanksgiving) are approximated — the LSTM
# training data comes from yfinance which only has actual trading days,
# so a slight mismatch here just shifts the prediction target by one day.
_FIXED_HOLIDAYS = {
    (1, 1),    # New Year's Day
    (6, 19),   # Juneteenth
    (7, 4),    # Independence Day
    (12, 25),  # Christmas Day
}


class DataStandardizer:
    ET = pytz.timezone("US/Eastern")

    @staticmethod
    def standardize_date(date_value) -> str:
        """Convert any date format into YYYY-MM-DD."""
        return pd.to_datetime(date_value).strftime("%Y-%m-%d")

    @classmethod
    def standardize_timestamp(cls, timestamp, to_timezone: str = "US/Eastern") -> str:
        """Convert a Unix timestamp or string to ISO-8601 in the given timezone."""
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        else:
            dt = pd.to_datetime(timestamp)
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
        return dt.astimezone(pytz.timezone(to_timezone)).isoformat()

    @classmethod
    def _next_trading_day(cls, d: date_type) -> date_type:
        """Advance past weekends and known US market holidays."""
        while d.weekday() >= 5 or (d.month, d.day) in _FIXED_HOLIDAYS:
            d += timedelta(days=1)
        return d

    @classmethod
    def apply_cutoff_rule(cls, published_at: str, cutoff_hour: int = 16) -> str:
        """
        Map a news publish time to the trading day it predicts.

        Before 4 PM ET on day T  →  predicts next trading day after T
        After  4 PM ET on day T  →  predicts next trading day after T+1

        Weekends and fixed US holidays are skipped automatically.
        """
        pub = pd.to_datetime(published_at)
        if pub.tzinfo is None:
            pub = cls.ET.localize(pub)
        else:
            pub = pub.astimezone(cls.ET)

        offset = 1 if pub.hour < cutoff_hour else 2
        target = pub.date() + timedelta(days=offset)
        target = cls._next_trading_day(target)
        return target.strftime("%Y-%m-%d")
