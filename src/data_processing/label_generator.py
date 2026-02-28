"""Generate up/down labels from price data.

For each (ticker, date) in the prices table, we look at the NEXT trading day's
close price. If it went up → label_binary = 1, down → label_binary = 0.

The last trading day for each ticker gets no label (no future data to compare).
Weekends/holidays are handled automatically because yfinance only returns
trading days — so "next row" IS the next trading day.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

from src.config import DATABASE_PATH, TICKERS

logger = logging.getLogger(__name__)


class LabelGenerator:
    def __init__(self, db_path: str | Path = DATABASE_PATH):
        self.db_path = Path(db_path)

    def generate(self, tickers: list[str] | None = None) -> dict:
        """
        Generate labels for all given tickers (default: all 15).

        Returns summary dict with counts per ticker.
        """
        tickers = tickers or TICKERS
        summary = {"tickers": {}, "total_labels": 0, "total_skipped": 0}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for ticker in tickers:
            added, skipped = self._generate_for_ticker(ticker, cursor)
            summary["tickers"][ticker] = {"labels": added, "skipped_dupes": skipped}
            summary["total_labels"] += added
            summary["total_skipped"] += skipped

        conn.commit()
        conn.close()

        logger.info("Generated %d labels total (%d duplicates skipped)",
                     summary["total_labels"], summary["total_skipped"])
        return summary

    def _generate_for_ticker(self, ticker: str, cursor: sqlite3.Cursor) -> tuple[int, int]:
        """
        For one ticker: grab prices sorted by date, pair each day with
        the next day, compute label, and insert into labels table.
        """
        # Get all trading days for this ticker, oldest first
        rows = cursor.execute(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date ASC",
            (ticker,),
        ).fetchall()

        if len(rows) < 2:
            logger.warning("%s: need at least 2 price rows, got %d", ticker, len(rows))
            return 0, 0

        added = 0
        skipped = 0

        # Walk through consecutive pairs: (day_t, day_t+1)
        for i in range(len(rows) - 1):
            date_t, close_t = rows[i]
            _, close_t1 = rows[i + 1]

            # 1 = stock went up, 0 = stock went down or stayed flat
            label_binary = 1 if close_t1 > close_t else 0

            # Percentage return: how much it moved
            pct_return = ((close_t1 - close_t) / close_t) * 100

            try:
                cursor.execute(
                    """INSERT INTO labels (ticker, date, label_binary, label_return, close_t, close_t_plus_1)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ticker, date_t, label_binary, round(pct_return, 4), close_t, close_t1),
                )
                added += 1
            except sqlite3.IntegrityError:
                # Already have a label for this (ticker, date) — skip
                skipped += 1

        logger.info("%s: %d labels generated, %d skipped (dupes)", ticker, added, skipped)
        return added, skipped
