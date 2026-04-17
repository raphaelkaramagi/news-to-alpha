"""Generate up/down labels from price data for 1-day and 3-day horizons.

For each (ticker, date) in the prices table, we look at the NEXT trading day's
close (1-day horizon) and 3 trading days later (3-day horizon).  Up = 1, down/flat = 0.

Horizon=3 uses close(t+3) vs close(t), so the last three trading days for each
ticker get NULL h3 labels.  Weekends/holidays are handled automatically because
yfinance only returns trading days — "next row" IS the next trading day.
"""

import sqlite3
import logging
from pathlib import Path

from src.config import DATABASE_PATH, TICKERS

logger = logging.getLogger(__name__)


class LabelGenerator:
    def __init__(self, db_path: str | Path = DATABASE_PATH):
        self.db_path = Path(db_path)

    def generate(self, tickers: list[str] | None = None) -> dict:
        """Generate labels for all given tickers (default: all 15).

        Writes both 1-day (label_binary / label_return / close_t_plus_1) and
        3-day (label_binary_h3 / return_h3 / close_t_plus_3) targets into the
        same row.  Returns a summary dict with counts per ticker.
        """
        tickers = tickers or TICKERS
        summary = {
            "tickers": {},
            "total_labels": 0,
            "total_updated": 0,
            "total_skipped": 0,
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for ticker in tickers:
            added, updated, skipped = self._generate_for_ticker(ticker, cursor)
            summary["tickers"][ticker] = {
                "labels": added,
                "updated": updated,
                "skipped_dupes": skipped,
            }
            summary["total_labels"] += added
            summary["total_updated"] += updated
            summary["total_skipped"] += skipped

        conn.commit()
        conn.close()

        logger.info(
            "Generated %d new labels (%d updated with h3, %d duplicates skipped)",
            summary["total_labels"], summary["total_updated"], summary["total_skipped"],
        )
        return summary

    def _generate_for_ticker(
        self, ticker: str, cursor: sqlite3.Cursor,
    ) -> tuple[int, int, int]:
        """Compute h1 + h3 labels for one ticker.

        Returns (added, updated, skipped) where `updated` counts rows where an
        existing (ticker, date) got its h3 columns back-filled.
        """
        rows = cursor.execute(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date ASC",
            (ticker,),
        ).fetchall()

        if len(rows) < 2:
            logger.warning("%s: need at least 2 price rows, got %d", ticker, len(rows))
            return 0, 0, 0

        added = 0
        updated = 0
        skipped = 0

        for i in range(len(rows) - 1):
            date_t, close_t = rows[i]
            _, close_t1 = rows[i + 1]

            label_binary = 1 if close_t1 > close_t else 0
            pct_return_1d = ((close_t1 - close_t) / close_t) * 100

            if i + 3 < len(rows):
                _, close_t3 = rows[i + 3]
                label_h3 = 1 if close_t3 > close_t else 0
                pct_return_3d = ((close_t3 - close_t) / close_t) * 100
            else:
                close_t3 = None
                label_h3 = None
                pct_return_3d = None

            try:
                cursor.execute(
                    """INSERT INTO labels (
                           ticker, date,
                           label_binary, label_return, close_t, close_t_plus_1,
                           label_binary_h3, return_h3, close_t_plus_3
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ticker, date_t,
                        label_binary, round(pct_return_1d, 4), close_t, close_t1,
                        label_h3,
                        round(pct_return_3d, 4) if pct_return_3d is not None else None,
                        close_t3,
                    ),
                )
                added += 1
            except sqlite3.IntegrityError:
                existing = cursor.execute(
                    "SELECT label_binary_h3 FROM labels WHERE ticker = ? AND date = ?",
                    (ticker, date_t),
                ).fetchone()
                if existing is not None and existing[0] is None and label_h3 is not None:
                    cursor.execute(
                        """UPDATE labels
                           SET label_binary_h3 = ?,
                               return_h3 = ?,
                               close_t_plus_3 = ?
                           WHERE ticker = ? AND date = ?""",
                        (
                            label_h3,
                            round(pct_return_3d, 4) if pct_return_3d is not None else None,
                            close_t3,
                            ticker, date_t,
                        ),
                    )
                    updated += 1
                else:
                    skipped += 1

        logger.info(
            "%s: %d new, %d updated (h3), %d duplicates",
            ticker, added, updated, skipped,
        )
        return added, updated, skipped
