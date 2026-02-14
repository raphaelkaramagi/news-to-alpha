#!/usr/bin/env python3
"""
Collect company news articles for the project tickers via Finnhub.

Usage:
    python scripts/collect_news.py                # default 21 days
    python scripts/collect_news.py --days 7
    python scripts/collect_news.py --tickers AAPL TSLA

Note: this may take a few minutes due to API rate limits (60 calls / min).
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS, FINNHUB_API_KEY  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_collection.news_collector import NewsCollector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect news articles")
    parser.add_argument(
        "--days", type=int, default=21,
        help="Days of news to collect (default: 21)",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Specific tickers (default: all from config)",
    )
    args = parser.parse_args()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    tickers = args.tickers or TICKERS

    # Ensure tables exist
    DatabaseSchema().create_all_tables()

    print("=" * 60)
    print("NEWS ARTICLE COLLECTION")
    print("=" * 60)
    print(f"Range   : {start_date}  ->  {end_date}")
    print(f"Tickers : {', '.join(tickers)}  ({len(tickers)} total)")
    print("Note: may take several minutes due to API rate limits\n")

    stats = NewsCollector(api_key=FINNHUB_API_KEY).collect(
        tickers, start_date, end_date
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Succeeded : {len(stats['tickers_succeeded'])} tickers")
    print(f"Failed    : {len(stats['tickers_failed'])} tickers")
    print(f"Articles  : {stats['rows_added']}")
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
