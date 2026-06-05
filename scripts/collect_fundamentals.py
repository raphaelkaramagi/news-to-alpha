#!/usr/bin/env python3
"""Collect free fundamentals + earnings dates from yfinance.

Populates two tables (created by src/database/schema.py):
  fundamentals    - sector, industry, market_cap, PE ratios, beta (one row/ticker)
  earnings_dates  - historical + upcoming earnings dates (for earnings-proximity)

These are free (yfinance) and feed:
  - earnings-proximity feature  (biggest free lever for the volatility model)
  - sector one-hot / sector-vol regime features
  - valuation ratios as slow-moving context

Usage
-----
  python scripts/collect_fundamentals.py
  python scripts/collect_fundamentals.py --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, TICKERS  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402


def _safe(info: dict, key: str) -> float | None:
    v = info.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def collect(tickers: list[str], db_path: Path = DATABASE_PATH) -> dict:
    import yfinance as yf

    DatabaseSchema(db_path).create_all_tables()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    stats = {"fundamentals": 0, "earnings_rows": 0, "failed": []}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = {}
            try:
                info = t.get_info() or {}
            except Exception:
                info = getattr(t, "info", {}) or {}

            cur.execute(
                """INSERT INTO fundamentals
                   (ticker, sector, industry, market_cap, trailing_pe, forward_pe, beta, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(ticker) DO UPDATE SET
                     sector=excluded.sector, industry=excluded.industry,
                     market_cap=excluded.market_cap, trailing_pe=excluded.trailing_pe,
                     forward_pe=excluded.forward_pe, beta=excluded.beta,
                     updated_at=datetime('now')""",
                (
                    ticker,
                    info.get("sector"),
                    info.get("industry"),
                    _safe(info, "marketCap"),
                    _safe(info, "trailingPE"),
                    _safe(info, "forwardPE"),
                    _safe(info, "beta"),
                ),
            )
            stats["fundamentals"] += 1

            # Earnings dates (past + upcoming). Prefer the historical table
            # (needs lxml); always also capture the next date from .calendar.
            found_dates: set[str] = set()
            try:
                ed = t.get_earnings_dates(limit=40)
                if ed is not None and not ed.empty:
                    for ts in ed.index:
                        try:
                            found_dates.add(ts.date().isoformat())
                        except Exception:
                            continue
            except Exception:
                pass
            try:
                cal = t.get_calendar() or {}
                for d in cal.get("Earnings Date", []) or []:
                    try:
                        found_dates.add(d.isoformat())
                    except Exception:
                        continue
            except Exception:
                pass

            for d in found_dates:
                cur.execute(
                    "INSERT OR IGNORE INTO earnings_dates (ticker, earnings_date) VALUES (?, ?)",
                    (ticker, d),
                )
                stats["earnings_rows"] += 1
            print(f"  {ticker}: sector={info.get('sector')!s:<22} "
                  f"earnings_dates+={len(found_dates)}")
        except Exception as exc:
            print(f"  {ticker}: FAILED — {exc}")
            stats["failed"].append(ticker)

    conn.commit()
    conn.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect free yfinance fundamentals + earnings dates")
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args()
    tickers = args.tickers or list(TICKERS)
    print(f"Collecting fundamentals for {len(tickers)} tickers ...")
    stats = collect(tickers)
    print(f"\nDone. fundamentals={stats['fundamentals']}  "
          f"earnings_rows={stats['earnings_rows']}  failed={stats['failed']}")


if __name__ == "__main__":
    main()
