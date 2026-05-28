#!/usr/bin/env python3
"""
Collect company news articles for the project tickers via Finnhub.

Usage:
    python scripts/collect_news.py                    # default 21 days
    python scripts/collect_news.py --days 365         # 1 year (Finnhub free tier max)
    python scripts/collect_news.py --tickers AAPL TSLA
    python scripts/collect_news.py --start-date 2024-01-01  # explicit backfill start

Backfill mode (accumulate historical news):
    python scripts/collect_news.py --days 365 --backfill

Repair sparse months (Finnhub drops mid-range days unless chunked):
    python scripts/collect_news.py --start-date 2026-05-01 --end-date 2026-05-27 --fill-gaps
    python scripts/collect_news.py --days 60 --fill-gaps --tickers AAPL NVDA

Note: may take several minutes due to API rate limits (60 calls / min).
Finnhub returns at most ~240 articles per API call; wide date ranges keep only
the newest articles. Always use 7-day chunks (default) or --fill-gaps for even
daily coverage.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS, FINNHUB_API_KEY  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_collection.news_collector import NewsCollector  # noqa: E402


def _date_chunks(start: str, end: str, chunk_days: int = 30) -> list[tuple[str, str]]:
    """Split [start, end] into inclusive windows of at most chunk_days.

    Finnhub's company-news endpoint returns ~200-250 articles per call. For
    wide date ranges it keeps only the newest articles in the window, so a
    single 365-day request often yields just the last few weeks. Monthly
    chunks are required to walk back through the full free-tier year.
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    if s > e:
        return []
    chunks: list[tuple[str, str]] = []
    cur = s
    while cur <= e:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), e)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _collect_range(
    collector: NewsCollector,
    ticker: str,
    start: str,
    end: str,
    chunk_days: int,
) -> dict:
    """Collect one ticker across one or more date chunks."""
    merged: dict = {
        "tickers_succeeded": [],
        "tickers_failed": [],
        "rows_added": 0,
        "duplicates_skipped": 0,
        "skipped_missing_fields": 0,
        "errors": {},
    }
    chunks = _date_chunks(start, end, chunk_days=chunk_days)
    if not chunks:
        return merged
    if len(chunks) > 1:
        print(f"    {ticker}: {len(chunks)} chunks ({chunks[0][0]} → {chunks[-1][1]})")
    for i, (c_start, c_end) in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(f"      chunk {i}/{len(chunks)}: {c_start} → {c_end}")
        s = collector.collect([ticker], c_start, c_end)
        for k in ("rows_added", "duplicates_skipped", "skipped_missing_fields"):
            merged[k] += s.get(k, 0)
        merged["tickers_succeeded"].extend(s.get("tickers_succeeded", []))
        merged["tickers_failed"].extend(s.get("tickers_failed", []))
        merged["errors"].update(s.get("errors", {}))
    return merged


def _missing_news_days(db_path: Path, ticker: str, start_date: str, end_date: str) -> list[str]:
    """Trading days in [start, end] with zero articles stored for this ticker."""
    import sqlite3

    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        trading = [
            r[0]
            for r in conn.execute(
                """
                SELECT date FROM prices
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
                """,
                (ticker, start_date, end_date),
            )
        ]
        have = {
            r[0]
            for r in conn.execute(
                """
                SELECT DISTINCT substr(published_at, 1, 10)
                FROM news
                WHERE ticker = ?
                  AND substr(published_at, 1, 10) BETWEEN ? AND ?
                """,
                (ticker, start_date, end_date),
            )
        }
    finally:
        conn.close()
    return [d for d in trading if d not in have]


def _collect_gap_days(
    collector: NewsCollector,
    ticker: str,
    days: list[str],
) -> dict:
    """Fetch one Finnhub call per missing trading day (best daily coverage)."""
    merged: dict = {
        "tickers_succeeded": [],
        "tickers_failed": [],
        "rows_added": 0,
        "duplicates_skipped": 0,
        "skipped_missing_fields": 0,
        "errors": {},
    }
    for day in days:
        s = collector.collect([ticker], day, day)
        for k in ("rows_added", "duplicates_skipped", "skipped_missing_fields"):
            merged[k] += s.get(k, 0)
        merged["tickers_succeeded"].extend(s.get("tickers_succeeded", []))
        merged["tickers_failed"].extend(s.get("tickers_failed", []))
        merged["errors"].update(s.get("errors", {}))
    return merged


def _earliest_news_date(db_path: Path, ticker: str) -> str | None:
    """Return the earliest news date in the DB for a ticker, or None if absent."""
    import sqlite3
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT MIN(published_at) FROM news WHERE ticker = ?", (ticker,)
        ).fetchone()
        return row[0][:10] if row and row[0] else None
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect news articles from Finnhub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days", type=int, default=21,
        help="Days of news to collect counting back from today (default: 21).",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Explicit ISO start date (YYYY-MM-DD) for backfill; overrides --days.",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="Explicit ISO end date (YYYY-MM-DD); defaults to today.",
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Specific tickers (default: all from config)",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help=(
            "Backfill mode: for each ticker, set end_date to the day before the "
            "earliest article already in the DB so we extend history without "
            "re-fetching recent news. Combine with --days 365 for maximum free-tier reach."
        ),
    )
    parser.add_argument(
        "--chunk-days", type=int, default=7,
        help=(
            "Split each ticker request into windows of this many days (default: 7). "
            "Finnhub returns at most ~240 articles per call and drops older rows "
            "when the range is too wide — never use a single call for multi-week ranges."
        ),
    )
    parser.add_argument(
        "--fill-gaps", action="store_true",
        help=(
            "After chunked collection, fetch any trading days in the range that still "
            "have zero stored articles (one API call per missing day). Use this to "
            "repair months where only the most recent days were saved."
        ),
    )
    args = parser.parse_args()

    end_dt = (
        datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date
        else datetime.now()
    )
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=args.days)

    end_date = end_dt.strftime("%Y-%m-%d")
    start_date = start_dt.strftime("%Y-%m-%d")
    tickers = args.tickers or TICKERS

    # Ensure tables exist
    DatabaseSchema().create_all_tables()

    from src.config import DATABASE_PATH  # noqa: E402

    print("=" * 60)
    print("NEWS ARTICLE COLLECTION")
    print("=" * 60)
    print(f"Range   : {start_date}  ->  {end_date}")
    print(f"Tickers : {', '.join(tickers)}  ({len(tickers)} total)")
    if args.backfill:
        print("Mode    : BACKFILL (per-ticker end adjusted to earliest existing date)")
    print(f"Chunks  : {args.chunk_days}-day windows (Finnhub caps ~240 articles/call)")
    if args.fill_gaps:
        print("Gaps    : per-day fetch for trading days still missing after chunks")
    print("Note: may take several minutes due to API rate limits\n")

    collector = NewsCollector(api_key=FINNHUB_API_KEY)
    total_stats: dict = {
        "tickers_succeeded": [],
        "tickers_failed": [],
        "rows_added": 0,
        "duplicates_skipped": 0,
        "skipped_missing_fields": 0,
        "errors": {},
    }

    for ticker in tickers:
        t_start, t_end = start_date, end_date
        if args.backfill:
            earliest = _earliest_news_date(DATABASE_PATH, ticker)
            if earliest:
                # Fetch the window just before what we already have
                backfill_end = (
                    datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if backfill_end <= t_start:
                    print(f"  {ticker}: already covered back to {earliest} — skipping")
                    total_stats["tickers_succeeded"].append(ticker)
                    continue
                t_end = backfill_end
                print(f"  {ticker}: backfilling {t_start} → {t_end} (earliest in DB: {earliest})")

        s = _collect_range(collector, ticker, t_start, t_end, chunk_days=args.chunk_days)
        for k in ("rows_added", "duplicates_skipped", "skipped_missing_fields"):
            total_stats[k] += s.get(k, 0)
        total_stats["tickers_succeeded"].extend(s.get("tickers_succeeded", []))
        total_stats["tickers_failed"].extend(s.get("tickers_failed", []))
        total_stats["errors"].update(s.get("errors", {}))

        if args.fill_gaps:
            missing = _missing_news_days(DATABASE_PATH, ticker, t_start, t_end)
            if missing:
                print(f"  {ticker}: filling {len(missing)} gap day(s): {missing[0]} .. {missing[-1]}")
                g = _collect_gap_days(collector, ticker, missing)
                for k in ("rows_added", "duplicates_skipped", "skipped_missing_fields"):
                    total_stats[k] += g.get(k, 0)
                total_stats["tickers_succeeded"].extend(g.get("tickers_succeeded", []))
                total_stats["tickers_failed"].extend(g.get("tickers_failed", []))
                total_stats["errors"].update(g.get("errors", {}))

    stats = total_stats

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Succeeded : {len(stats['tickers_succeeded'])} tickers")
    print(f"Failed    : {len(stats['tickers_failed'])} tickers")
    print(f"Articles  : {stats['rows_added']}")
    print(f"Duplicates: {stats['duplicates_skipped']}")
    print(f"Skipped (missing fields): {stats.get('skipped_missing_fields', 0)}")

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
