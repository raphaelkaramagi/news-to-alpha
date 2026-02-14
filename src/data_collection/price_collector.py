"""Collect stock prices from Yahoo Finance."""

import json
import sqlite3
import time
import logging
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

from src.data_collection.base_collector import BaseCollector
from src.config import DATABASE_PATH, MAX_RETRIES, RETRY_BASE_DELAY_SECONDS

logger = logging.getLogger(__name__)


class PriceCollector(BaseCollector):
    def __init__(self, db_path: str = str(DATABASE_PATH)):
        super().__init__(db_path)

    def collect(self, tickers: list[str], start_date: str, end_date: str, **kwargs) -> dict:
        """Download price data and insert into database."""
        max_retries = kwargs.get("max_retries", MAX_RETRIES)
        stats = {
            "tickers_succeeded": [],
            "tickers_failed": [],
            "rows_added": 0,
            "duplicates_skipped": 0,
            "errors": {},
        }

        run_start = datetime.now()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for ticker in tickers:
            added, dupes, error = self._collect_one(ticker, start_date, end_date, cursor, max_retries)
            stats["rows_added"] += added
            stats["duplicates_skipped"] += dupes

            if error:
                stats["tickers_failed"].append(ticker)
                stats["errors"][ticker] = error
            else:
                stats["tickers_succeeded"].append(ticker)

        conn.commit()
        conn.close()

        duration = (datetime.now() - run_start).total_seconds()
        self._log_run(stats, run_start, duration)
        return stats

    def _collect_one(self, ticker: str, start_date: str, end_date: str, 
                     cursor: sqlite3.Cursor, max_retries: int) -> tuple[int, int, str | None]:
        """Fetch and insert data for one ticker with retry logic."""
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info("Fetching %s  %s -> %s  (attempt %d/%d)", 
                               ticker, start_date, end_date, attempt, max_retries)
                
                df = yf.download(ticker, start=start_date, end=end_date, 
                               progress=False, auto_adjust=False)

                if df.empty:
                    return 0, 0, "No data returned"

                added, dupes = self._insert_rows(ticker, df, cursor)
                self.logger.info("%s: +%d rows, %d duplicates", ticker, added, dupes)
                return added, dupes, None

            except Exception as exc:
                if attempt < max_retries:
                    wait = RETRY_BASE_DELAY_SECONDS ** attempt
                    self.logger.warning("%s attempt %d failed - retrying in %.1fs", ticker, attempt, wait)
                    time.sleep(wait)
                else:
                    self.logger.error("%s failed after %d attempts: %s", ticker, max_retries, exc)
                    return 0, 0, str(exc)

        return 0, 0, "unexpected"

    @staticmethod
    def _insert_rows(ticker: str, df: pd.DataFrame, cursor: sqlite3.Cursor) -> tuple[int, int]:
        """Insert DataFrame rows into database."""
        added = 0
        dupes = 0

        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel("Ticker", axis=1)

        for date_idx, row in df.iterrows():
            try:
                cursor.execute(
                    """INSERT INTO prices (ticker, date, open, high, low, close, volume, adjusted_close)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (ticker, pd.Timestamp(date_idx).strftime("%Y-%m-%d"),
                     float(row["Open"]), float(row["High"]), float(row["Low"]),
                     float(row["Close"]), int(row["Volume"]),
                     float(row["Adj Close"]) if "Adj Close" in row.index else None)
                )
                added += 1
            except sqlite3.IntegrityError:
                dupes += 1
            except Exception as exc:
                logger.debug("Row insert error: %s", exc)

        return added, dupes

    def _log_run(self, stats: dict, started: datetime, duration: float) -> None:
        """Save run info to run_log table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO run_log 
               (run_type, status, tickers_attempted, tickers_succeeded, tickers_failed, 
                rows_added, duplicates_skipped, error_message, started_at, completed_at, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("price_collection", "success" if not stats["tickers_failed"] else "partial",
             json.dumps(stats["tickers_succeeded"] + stats["tickers_failed"]),
             json.dumps(stats["tickers_succeeded"]), json.dumps(stats["tickers_failed"]),
             stats["rows_added"], stats["duplicates_skipped"],
             json.dumps(stats["errors"]) if stats["errors"] else None,
             started.isoformat(), datetime.now().isoformat(), round(duration, 2))
        )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.database.schema import DatabaseSchema
    DatabaseSchema().create_all_tables()

    collector = PriceCollector()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    print(f"\nTest: collecting AAPL  {start} -> {end}\n")
    result = collector.collect(["AAPL"], start, end)
    print(f"\nRows added: {result['rows_added']}, Duplicates: {result['duplicates_skipped']}")
