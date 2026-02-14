"""Check price data for missing values, anomalies, and gaps."""

import sqlite3
import logging
import pandas as pd
from src.config import DATABASE_PATH, TICKERS

logger = logging.getLogger(__name__)


class PriceDataValidator:
    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.db_path = db_path

    def validate(self, tickers: list[str] | None = None) -> dict:
        tickers = tickers or TICKERS
        conn = sqlite3.connect(self.db_path)
        results = {
            "missing_data": self._check_missing(conn, tickers),
            "price_anomalies": self._check_price_jumps(conn, tickers),
            "volume_anomalies": self._check_zero_volume(conn, tickers),
            "coverage": self._check_coverage(conn, tickers),
        }
        conn.close()
        return results

    def _check_missing(self, conn, tickers):
        """Rows where any OHLCV field is NULL."""
        ph = ",".join(["?"] * len(tickers))
        df = pd.read_sql(f"""
            SELECT ticker, COUNT(*) AS null_count FROM prices
            WHERE ticker IN ({ph})
              AND (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL)
            GROUP BY ticker
        """, conn, params=tickers)
        if not df.empty:
            logger.warning("Tickers with missing values: %d", len(df))
        return df.to_dict("records")

    def _check_price_jumps(self, conn, tickers):
        """Daily close-to-close moves > 20%."""
        ph = ",".join(["?"] * len(tickers))
        df = pd.read_sql(f"""
            WITH changes AS (
                SELECT ticker, date, close,
                       LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev,
                       ((close - LAG(close) OVER (PARTITION BY ticker ORDER BY date))
                        / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) * 100 AS pct_change
                FROM prices WHERE ticker IN ({ph})
            )
            SELECT ticker, date, close, prev, ROUND(pct_change, 2) AS pct_change
            FROM changes WHERE ABS(pct_change) > 20
        """, conn, params=tickers)
        if not df.empty:
            logger.warning("Suspicious price jumps: %d", len(df))
        return df.to_dict("records")

    def _check_zero_volume(self, conn, tickers):
        ph = ",".join(["?"] * len(tickers))
        df = pd.read_sql(f"SELECT ticker, date, volume FROM prices WHERE ticker IN ({ph}) AND volume = 0", conn, params=tickers)
        if not df.empty:
            logger.warning("Zero-volume days: %d", len(df))
        return df.to_dict("records")

    def _check_coverage(self, conn, tickers):
        """Per-ticker row counts and date ranges."""
        ph = ",".join(["?"] * len(tickers))
        return pd.read_sql(f"""
            SELECT ticker, COUNT(DISTINCT date) AS days_collected, MIN(date) AS first_date, MAX(date) AS last_date
            FROM prices WHERE ticker IN ({ph}) GROUP BY ticker
        """, conn, params=tickers).to_dict("records")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = PriceDataValidator().validate()
    print("\n=== PRICE DATA VALIDATION ===")
    print(f"Missing values : {len(r['missing_data'])}")
    print(f"Price anomalies: {len(r['price_anomalies'])}")
    print(f"Zero-volume    : {len(r['volume_anomalies'])}")
    for row in r["coverage"]:
        print(f"  {row['ticker']:5s}  {row['days_collected']:3d} days  ({row['first_date']} -> {row['last_date']})")
