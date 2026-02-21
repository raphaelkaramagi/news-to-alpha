"""Split data chronologically into train / validation / test sets.

Since this is time series data, If we split randomly, the model
could train on February data and predict January. 

Default split: 70% train / 15% validation / 15% test (by date, not row count).
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class DatasetSplitter:
    def __init__(self, db_path: str | Path = DATABASE_PATH):
        self.db_path = Path(db_path)

    def split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        # test gets whatever is left (1 - train - val)
    ) -> dict:
        """
        Split all data by date into train/val/test.

        Returns a summary dict with date ranges and row counts per split.
        Also saves the split mapping to data/processed/split_info.json.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Get every unique trading date across all tickers, sorted
        all_dates = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT date FROM prices ORDER BY date"
            ).fetchall()
        ]

        if len(all_dates) < 3:
            conn.close()
            raise ValueError(f"Need at least 3 trading dates to split, got {len(all_dates)}")

        # Figure out where to cut
        n = len(all_dates)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Make sure each split has at least 1 date
        train_end = max(train_end, 1)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, n - 1)  # leave at least 1 for test

        train_dates = all_dates[:train_end]
        val_dates = all_dates[train_end:val_end]
        test_dates = all_dates[val_end:]

        # Count rows in each split for prices, news, and labels
        summary = {
            "train": self._count_split(conn, train_dates),
            "val": self._count_split(conn, val_dates),
            "test": self._count_split(conn, test_dates),
        }

        conn.close()

        # Save to data/processed/split_info.json
        self._save_split_info(summary)

        return summary

    def _count_split(self, conn: sqlite3.Connection, dates: list[str]) -> dict:
        """Count how many rows from each table fall in this date range."""
        if not dates:
            return {"dates": [], "date_range": None, "prices": 0, "news": 0, "labels": 0}

        placeholders = ",".join("?" for _ in dates)

        prices = conn.execute(
            f"SELECT COUNT(*) FROM prices WHERE date IN ({placeholders})", dates
        ).fetchone()[0]

        # News uses published_at (ISO timestamp), so match on the date portion
        news = conn.execute(
            f"SELECT COUNT(*) FROM news WHERE substr(published_at, 1, 10) IN ({placeholders})",
            dates,
        ).fetchone()[0]

        labels = conn.execute(
            f"SELECT COUNT(*) FROM labels WHERE date IN ({placeholders})", dates
        ).fetchone()[0]

        return {
            "dates": dates,
            "date_range": f"{dates[0]} to {dates[-1]}",
            "num_days": len(dates),
            "prices": prices,
            "news": news,
            "labels": labels,
        }

    def _save_split_info(self, summary: dict) -> None:
        """Write the split details to data/processed/split_info.json."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DATA_DIR / "split_info.json"

        # Build a clean version (no huge date lists in the top-level view)
        output = {
            "created_at": datetime.now().isoformat(),
            "splits": {},
        }
        for split_name, info in summary.items():
            output["splits"][split_name] = {
                "date_range": info["date_range"],
                "num_days": info["num_days"],
                "prices": info["prices"],
                "news": info["news"],
                "labels": info["labels"],
                "dates": info["dates"],
            }

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info("Split info saved to %s", out_path)
