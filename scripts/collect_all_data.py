#!/usr/bin/env python3
"""
One-command data pipeline: collect prices + news, then validate.

Usage:
    python scripts/collect_all_data.py
    python scripts/collect_all_data.py --days 30
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS, FINNHUB_API_KEY  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_collection.price_collector import PriceCollector  # noqa: E402
from src.data_collection.news_collector import NewsCollector  # noqa: E402
from src.data_processing.price_validation import PriceDataValidator  # noqa: E402
from src.data_processing.news_validation import NewsDataValidator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Full data collection pipeline")
    parser.add_argument("--days", type=int, default=21)
    args = parser.parse_args()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print("=" * 70)
    print("COMPLETE DATA COLLECTION PIPELINE")
    print("=" * 70)
    print(f"Range   : {start_date}  ->  {end_date}")
    print(f"Tickers : {len(TICKERS)} total\n")

    # 0) Ensure DB exists
    DatabaseSchema().create_all_tables()

    # 1) Prices
    print("--- Step 1: Price data ---")
    price_stats = PriceCollector().collect(TICKERS, start_date, end_date)
    print(f"  Rows added: {price_stats['rows_added']}\n")

    # 2) News
    print("--- Step 2: News articles ---")
    if FINNHUB_API_KEY:
        news_stats = NewsCollector(api_key=FINNHUB_API_KEY).collect(
            TICKERS, start_date, end_date
        )
        print(f"  Articles added: {news_stats['rows_added']}\n")
    else:
        print("  Skipped (no API key — set NEWS_API_KEY in .env)\n")
        news_stats = {"rows_added": 0, "duplicates_skipped": 0,
                      "tickers_succeeded": [], "tickers_failed": [], "errors": {}}

    # 3) Validate
    print("--- Step 3: Validation ---")
    pv = PriceDataValidator().validate(TICKERS)
    nv = NewsDataValidator().validate(TICKERS)

    price_issues = (
        len(pv["missing_data"])
        + len(pv["price_anomalies"])
        + len(pv["volume_anomalies"])
    )
    # Support both naming conventions (old + new)
    dup_key = "duplicate_url_ticker_pairs" if "duplicate_url_ticker_pairs" in nv else "duplicate_urls"

    news_issues = (
        len(nv["missing_fields"])
        + len(nv["future_timestamps"])
        + len(nv[dup_key])
    )

    print(f"  Price issues : {price_issues}")
    print(f"  News issues  : {news_issues}")

    # Summary
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Prices  -> {price_stats['rows_added']} new rows  "
          f"({price_stats['duplicates_skipped']} dupes skipped)")
    print(f"News    -> {news_stats['rows_added']} new articles  "
          f"({news_stats['duplicates_skipped']} dupes skipped)")

    if price_stats["tickers_failed"]:
        print(f"\nFailed price tickers: {', '.join(price_stats['tickers_failed'])}")
    if news_stats["tickers_failed"]:
        print(f"Failed news tickers : {', '.join(news_stats['tickers_failed'])}")

    print()


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    main()
