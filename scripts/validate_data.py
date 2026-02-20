#!/usr/bin/env python3
"""
Run data-quality checks on both price and news data.

Usage:
    python scripts/validate_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS  # noqa: E402
from src.data_processing.price_validation import PriceDataValidator  # noqa: E402
from src.data_processing.news_validation import NewsDataValidator  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    # ---- Price checks ------------------------------------------------
    pv = PriceDataValidator().validate(TICKERS)

    print("\n--- PRICE DATA ---")
    print(f"Missing values : {len(pv['missing_data'])} tickers affected")
    print(f"Price anomalies: {len(pv['price_anomalies'])} events")
    print(f"Zero-volume    : {len(pv['volume_anomalies'])} days")

    print("\nCoverage:")
    for row in pv["coverage"]:
        print(f"  {row['ticker']:5s}  {row['days_collected']:3d} days  "
              f"({row['first_date']} -> {row['last_date']})")

    # ---- News checks -------------------------------------------------
    nv = NewsDataValidator().validate(TICKERS)

    print("\n--- NEWS DATA ---")
    print(f"Missing fields   : {len(nv['missing_fields'])} tickers affected")
    print(f"Future timestamps: {len(nv['future_timestamps'])} articles")
    print(f"Duplicate (url,ticker) pairs: {len(nv['duplicate_url_ticker_pairs'])} duplicates")
    print("\nArticle distribution:")
    for row in nv["articles_per_ticker"]:
        print(f"  {row['ticker']:5s}  {row['article_count']:4d} articles")

    # ---- Overall verdict ---------------------------------------------
    total_issues = (
        len(pv["missing_data"])
        + len(pv["price_anomalies"])
        + len(pv["volume_anomalies"])
        + len(nv["missing_fields"])
        + len(nv["future_timestamps"])
        + len(nv["duplicate_url_ticker_pairs"])    )

    print("\n" + "=" * 60)
    if total_issues == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"TOTAL ISSUES: {total_issues}  (review above for details)")
    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
