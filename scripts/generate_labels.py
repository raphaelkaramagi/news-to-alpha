#!/usr/bin/env python3
"""
Generate up/down labels from price data and populate the labels table.

Definition:
    labels.date       = prediction_date (the day we predict)
    label_binary      = 1 if close(prediction_date) > close(previous trading day), else 0
    label_return      = percent change between the two closes
    close_t           = close on the previous trading day
    close_t_plus_1    = close on prediction_date

Usage:
    python scripts/generate_labels.py
    python scripts/generate_labels.py --tickers AAPL TSLA
"""

import sys
import argparse
import sqlite3
from pathlib import Path

# Add project root to path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS, DATABASE_PATH  # noqa: E402


def main() -> None:
    # ── CLI arguments ────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Generate binary labels from price data")
    parser.add_argument("--tickers", nargs="+", help="Tickers to process (default: all 15)")
    args = parser.parse_args()

    # Fall back to the full ticker list from config if none specified
    tickers = args.tickers or TICKERS

    # ── Database connection ──────────────────────────────────
    db = Path(DATABASE_PATH)
    
    conn = sqlite3.connect(db)

    # Running totals across all tickers
    total_created = 0
    total_skipped = 0

    # ── Process each ticker ──────────────────────────────────
    for ticker in tickers:

        # Query every (date, close) pair for this ticker, sorted oldest → newest.
        # We need chronological order so rows[i-1] is always the previous trading day.
        rows = conn.execute(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
            (ticker,),
        ).fetchall()

        # Need at least 2 rows to compare consecutive days
        if len(rows) < 2:
            print(f"  {ticker:5s}  skipped (fewer than 2 price rows)")
            continue

        created = 0   # new labels inserted this ticker
        skipped = 0   # labels that already existed (duplicate ticker+date)

        # Loop from the second row onward so we always have a "previous day" to compare
        for i in range(1, len(rows)):

            # Previous trading day's date and closing price
            prev_date, prev_close = rows[i - 1]

            # Prediction date (the day we're generating a label for) and its closing price
            pred_date, pred_close = rows[i]

            # ── Binary label ─────────────────────────────────
            # 1 = stock went UP from previous close to today's close
            # 0 = stock went DOWN or stayed flat
            label_binary = 1 if pred_close > prev_close else 0

            # ── Percent return ───────────────────────────────
            # How much the price moved as a percentage (positive = up, negative = down)
            label_return = (pred_close - prev_close) / prev_close * 100 if prev_close else 0.0

            # ── Insert into the labels table ─────────────────
            # The table has a UNIQUE(ticker, date) constraint, so if this label
            # already exists we catch the IntegrityError and count it as skipped.
            # Columns stored:
            #   close_t        = previous day's close  (the "before" price)
            #   close_t_plus_1 = prediction day's close (the "after" price)
            try:
                conn.execute(
                    """INSERT INTO labels
                           (ticker, date, label_binary, label_return, close_t, close_t_plus_1)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ticker, pred_date, label_binary, label_return, prev_close, pred_close),
                )
                created += 1
            except sqlite3.IntegrityError:
                # Label for this ticker+date already exists — skip it
                skipped += 1

        # ── Per-ticker summary ───────────────────────────────
        print(f"  {ticker:5s}  {created} created, {skipped} skipped (dupes)")
        total_created += created
        total_skipped += skipped

    # ── Commit all inserts at once and close ─────────────────
    conn.commit()
    conn.close()

    # ── Final summary ────────────────────────────────────────
    print(f"  Total: {total_created} labels created, {total_skipped} skipped")


if __name__ == "__main__":
    main()
