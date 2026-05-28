#!/usr/bin/env python3
"""
Collect historical stock prices for the project tickers.

Usage:
    python scripts/collect_prices.py                # default 21 days
    python scripts/collect_prices.py --days 60      # last 60 days
    python scripts/collect_prices.py --tickers AAPL TSLA
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS, MARKET_INDEX_TICKER  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_collection.price_collector import PriceCollector  # noqa: E402

# VIX is collected as a regime feature alongside SPY. yfinance uses "^VIX"
# but we store it in the DB as "VIX" (see PriceCollector._collect_one).
VIX_TICKER = "VIX"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect stock price data")
    parser.add_argument(
        "--days", type=int, default=365,
        help="Days of history to collect (default: 365). Ignored if --since or --incremental.",
    )
    parser.add_argument(
        "--since", default=None,
        help="ISO start date (YYYY-MM-DD). Overrides --days.",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Start from DB coverage (min per-ticker latest price) with buffer.",
    )
    parser.add_argument(
        "--buffer-days", type=int, default=5,
        help="Overlap days when using --incremental (default: 5).",
    )
    parser.add_argument(
        "--min-days", type=int, default=7,
        help="Minimum calendar lookback when using --incremental (default: 7).",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Specific tickers (default: all from config)",
    )
    args = parser.parse_args()

    end_date = datetime.now().strftime("%Y-%m-%d")
    tickers = args.tickers or TICKERS
    # Always include the market-index and VIX tickers for LSTM regime features.
    for regime_ticker in (MARKET_INDEX_TICKER, VIX_TICKER):
        if regime_ticker not in tickers:
            tickers = [*tickers, regime_ticker]

    if args.since:
        start_date = args.since
        window_mode = "explicit"
    elif args.incremental:
        from src.utils.collection_window import compute_collection_window  # noqa: E402
        start_date, end_date, win = compute_collection_window(
            tickers,
            buffer_days=args.buffer_days,
            min_days=args.min_days,
            max_days=args.days,
        )
        window_mode = win["mode"]
    else:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        window_mode = "fixed_days"

    # Ensure tables exist
    DatabaseSchema().create_all_tables()

    print("=" * 60)
    print("STOCK PRICE DATA COLLECTION")
    print("=" * 60)
    print(f"Mode    : {window_mode}")
    print(f"Range   : {start_date}  ->  {end_date}")
    print(f"Tickers : {', '.join(tickers)}  ({len(tickers)} total)\n")

    stats = PriceCollector().collect(tickers, start_date, end_date)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Succeeded : {len(stats['tickers_succeeded'])} tickers")
    print(f"Failed    : {len(stats['tickers_failed'])} tickers")
    print(f"Rows added: {stats['rows_added']}")
    print(f"Duplicates: {stats['duplicates_skipped']}")

    if stats["errors"]:
        print("\nErrors:")
        for t, e in stats["errors"].items():
            print(f"  {t}: {e}")

    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    main()
