#!/usr/bin/env python3
"""
Quick demo showing the entire pipeline working.
Run: python scripts/demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta
from src.database.schema import DatabaseSchema
from src.data_collection.price_collector import PriceCollector
from src.data_collection.news_collector import NewsCollector
from src.data_processing.price_validation import PriceDataValidator
from src.config import FINNHUB_API_KEY, DATABASE_PATH
import sqlite3

def main():
    print("=" * 70)
    print("STOCK PREDICTION PLATFORM - DEMO")
    print("=" * 70)
    
    db_path = str(DATABASE_PATH)

    # Setup
    print("\n1. Setting up database...")
    DatabaseSchema(db_path).create_all_tables()
    print("   ✓ Database ready with 5 tables")
    
    # Dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    test_tickers = ["AAPL", "TSLA"]
    
    # Collect prices
    print(f"\n2. Collecting prices for {', '.join(test_tickers)}...")
    print(f"   Date range: {start_date} to {end_date}")
    price_stats = PriceCollector(db_path).collect(test_tickers, start_date, end_date)
    print(f"   ✓ Added {price_stats['rows_added']} price rows")
    
    # Collect news
    if FINNHUB_API_KEY:
        print(f"\n3. Collecting news for {test_tickers[0]}...")
        news_stats = NewsCollector(FINNHUB_API_KEY, db_path).collect(
            [test_tickers[0]], start_date, end_date
        )
        print(f"   ✓ Added {news_stats['rows_added']} news articles")
    else:
        print("\n3. Skipping news (no API key in .env)")
    
    # Validate
    print("\n4. Validating data quality...")
    validator = PriceDataValidator(db_path)
    results = validator.validate(test_tickers)
    
    for ticker_info in results["coverage"]:
        print(f"   {ticker_info['ticker']}: {ticker_info['days_collected']} days "
              f"({ticker_info['first_date']} to {ticker_info['last_date']})")
    
    # Show sample data
    print("\n5. Sample data:")
    conn = sqlite3.connect(db_path)
    
    # Sample prices
    cursor = conn.execute(
        "SELECT ticker, date, close FROM prices WHERE ticker IN (?, ?) ORDER BY date DESC LIMIT 5",
        test_tickers
    )
    print("\n   Recent prices:")
    for row in cursor:
        print(f"   {row[0]:5s} {row[1]}  ${row[2]:.2f}")
    
    # Sample news (if any)
    cursor = conn.execute(
        "SELECT ticker, title FROM news WHERE ticker = ? LIMIT 3",
        (test_tickers[0],)
    )
    news_rows = cursor.fetchall()
    if news_rows:
        print(f"\n   Recent {test_tickers[0]} headlines:")
        for row in news_rows:
            title = row[1][:60] + "..." if len(row[1]) > 60 else row[1]
            print(f"   • {title}")
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  • Check database: sqlite3 data/database.db")
    print("  • Run full collection: python scripts/collect_all_data.py")
    print("  • Run tests: pytest tests/ -v")
    print()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Keep output clean
    main()
