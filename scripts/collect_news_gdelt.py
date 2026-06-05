#!/usr/bin/env python3
"""Extend news history with free, keyless GDELT (DOC 2.0 API).

Finnhub free only gives ~6 months of company news, which is the binding
constraint for honestly validating the news models. GDELT DOC 2.0 is free,
keyless, and covers years of global news. This collector queries by company
name in monthly windows and inserts headlines into the same `news` table, then
you run scripts/backfill_news_sentiment.py to score them.

GDELT gives title + url + domain + seendate (no summary), so `content` is left
empty and sentiment is computed from titles.

IMPORTANT — gating: only adopt this extended history if the walk-forward harness
(or news-model test AUC) actually improves. GDELT is noisier than Finnhub.

Usage
-----
  python scripts/collect_news_gdelt.py --months-back 24
  python scripts/collect_news_gdelt.py --tickers AAPL MSFT --months-back 12
  python scripts/collect_news_gdelt.py --dry-run --tickers AAPL --months-back 2
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, TICKER_TO_COMPANY, TICKERS  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _month_windows(months_back: int) -> list[tuple[datetime, datetime]]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    windows = []
    cur_end = now
    for _ in range(months_back):
        cur_start = cur_end - timedelta(days=30)
        windows.append((cur_start, cur_end))
        cur_end = cur_start
    return windows


def _fetch_window(query: str, start: datetime, end: datetime, max_records: int) -> list[dict]:
    import requests

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "DateAsc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for attempt in range(3):
        try:
            resp = requests.get(
                GDELT_URL, params=params, timeout=45,
                headers={"User-Agent": "news-to-alpha/1.0"},
            )
            if resp.status_code != 200 or not resp.text.strip():
                return []
            try:
                data = resp.json()
            except ValueError:
                return []
            return data.get("articles", []) or []
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def _parse_seendate(s: str) -> str | None:
    """GDELT seendate like '20240115T123000Z' -> ISO date-time."""
    try:
        dt = datetime.strptime(s[:15], "%Y%m%dT%H%M%S")
        return dt.isoformat()
    except Exception:
        return None


def collect(
    tickers: list[str],
    months_back: int,
    *,
    max_per_window: int = 75,
    dry_run: bool = False,
    db_path: Path = DATABASE_PATH,
) -> dict:
    DatabaseSchema(db_path).create_all_tables()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    stats = {"added": 0, "dupes": 0, "tickers": {}}
    windows = _month_windows(months_back)

    for ticker in tickers:
        company = TICKER_TO_COMPANY.get(ticker, ticker)
        query = f'"{company}" sourcelang:english'
        added_t = 0
        for (start, end) in windows:
            arts = _fetch_window(query, start, end, max_per_window)
            for a in arts:
                url = (a.get("url") or "").strip()
                title = (a.get("title") or "").strip()
                source = (a.get("domain") or "gdelt").strip()
                pub = _parse_seendate(a.get("seendate", ""))
                if not url or not title or not pub:
                    continue
                if dry_run:
                    added_t += 1
                    continue
                try:
                    cur.execute(
                        """INSERT INTO news (url, ticker, title, source, published_at, content)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (url, ticker, title, source, pub, ""),
                    )
                    added_t += 1
                except sqlite3.IntegrityError:
                    stats["dupes"] += 1
            time.sleep(0.5)  # be polite to the free endpoint
        stats["added"] += added_t
        stats["tickers"][ticker] = added_t
        print(f"  {ticker}: +{added_t} headlines ({months_back} monthly windows)")
        if not dry_run:
            conn.commit()

    conn.commit()
    conn.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Extend news history via free GDELT")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--months-back", type=int, default=24)
    ap.add_argument("--max-per-window", type=int, default=75)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tickers = args.tickers or list(TICKERS)
    print(f"GDELT news backfill: {len(tickers)} tickers x {args.months_back} months ...")
    stats = collect(
        tickers, args.months_back,
        max_per_window=args.max_per_window, dry_run=args.dry_run,
    )
    print(f"\nDone. added={stats['added']}  dupes={stats['dupes']}")
    if not args.dry_run:
        print("Next: python scripts/backfill_news_sentiment.py  (score the new headlines)")


if __name__ == "__main__":
    main()
