"""Check news data for missing fields, bad timestamps, and duplicates."""

import sqlite3
import logging
from datetime import datetime
import pandas as pd
import pytz
from src.config import DATABASE_PATH, TICKERS

logger = logging.getLogger(__name__)


class NewsDataValidator:
    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.db_path = db_path

    def validate(self, tickers: list[str] | None = None) -> dict:
        tickers = tickers or TICKERS
        conn = sqlite3.connect(self.db_path)
        results = {
            "missing_fields": self._check_missing_fields(conn, tickers),
            "future_timestamps": self._check_future_timestamps(conn),
            "duplicate_url_ticker_pairs": self._check_duplicate_url_ticker_pairs(conn),
            "articles_per_ticker": self._check_distribution(conn, tickers),
        }
        conn.close()
        return results

    def _check_missing_fields(self, conn, tickers):
        ph = ",".join(["?"] * len(tickers))
        df = pd.read_sql(f"""
            SELECT ticker, COUNT(*) AS missing_count FROM news
            WHERE ticker IN ({ph})
              AND (title IS NULL OR title = '' OR published_at IS NULL OR source IS NULL OR source = '')
            GROUP BY ticker
        """, conn, params=tickers)
        if not df.empty:
            logger.warning("Tickers with missing fields: %d", len(df))
        return df.to_dict("records")

    def _check_future_timestamps(self, conn, buffer_minutes: int = 10):
        """
        Flag articles whose published_at is truly in the future.

        We parse published_at as a real timezone-aware datetime (UTC) to avoid
        incorrect string comparisons.
        """
        df = pd.read_sql(
            "SELECT ticker, title, published_at FROM news WHERE published_at IS NOT NULL",
            conn,
        )
        if df.empty:
            return []

        # Parse ISO-8601 timestamps with timezone offsets safely
        ts = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

        # If parsing fails, treat as an issue (optional: include in missing_fields instead)
        bad_parse = df[ts.isna()].copy()
        bad_parse_records = bad_parse.to_dict("records") if not bad_parse.empty else []

        now_utc = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=buffer_minutes)

        future = df[ts > now_utc].copy()
        future_records = future.to_dict("records")

        if not future.empty:
            logger.warning("Articles with future timestamps: %d", len(future))

        # If you want to track bad parses too, return them combined.
        # Otherwise, return only future_records.
        return future_records

    def _check_duplicate_url_ticker_pairs(self, conn):
        df = pd.read_sql("""
            SELECT url, ticker, COUNT(*) AS cnt
            FROM news
            GROUP BY url, ticker
            HAVING cnt > 1
        """, conn)
        if not df.empty:
            logger.warning("Duplicate (url, ticker) pairs: %d", len(df))
        return df.to_dict("records")

    def _check_distribution(self, conn, tickers):
        ph = ",".join(["?"] * len(tickers))
        return pd.read_sql(f"""
            SELECT ticker, COUNT(*) AS article_count, MIN(published_at) AS earliest, MAX(published_at) AS latest
            FROM news WHERE ticker IN ({ph}) GROUP BY ticker ORDER BY article_count DESC
        """, conn, params=tickers).to_dict("records")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = NewsDataValidator().validate()
    print("\n=== NEWS DATA VALIDATION ===")
    print(f"Missing fields : {len(r['missing_fields'])}")
    print(f"Future timestamps: {len(r['future_timestamps'])}")
    print(f"Duplicate (url, ticker) pairs : {len(r['duplicate_url_ticker_pairs'])}")
    for row in r["articles_per_ticker"]:
        print(f"  {row['ticker']:5s}  {row['article_count']:4d} articles")
