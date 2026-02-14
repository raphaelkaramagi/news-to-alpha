"""Date/time standardization and cutoff-time logic."""

from datetime import datetime, timedelta
import pandas as pd
import pytz


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
    def apply_cutoff_rule(cls, published_at: str, cutoff_hour: int = 16) -> str:
        """
        Map a news publish time to the trading day it predicts.
        Before 4 PM ET on day T -> predicts T+1
        After  4 PM ET on day T -> predicts T+2
        (Weekend/holiday adjustment deferred to Week 3)
        """
        pub = pd.to_datetime(published_at)
        if pub.tzinfo is None:
            pub = cls.ET.localize(pub)
        else:
            pub = pub.astimezone(cls.ET)

        offset = 1 if pub.hour < cutoff_hour else 2
        return (pub.date() + timedelta(days=offset)).strftime("%Y-%m-%d")
