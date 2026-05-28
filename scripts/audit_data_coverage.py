#!/usr/bin/env python3
"""Audit price, news, label, and live-row consistency across the pipeline.

Reports:
  - Price coverage: min/max per ticker, gaps in last N trading days
  - News coverage: trading days with zero cutoff-aligned headlines per ticker/month
  - Label lag vs price lag
  - Live rows where has_news=0 but n_headlines>0 (should be zero after Phase 1 fix)

Usage
-----
  python scripts/audit_data_coverage.py
  python scripts/audit_data_coverage.py --days 365
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR, TICKERS  # noqa: E402
from src.models.news_pipeline import _load_news_aligned  # noqa: E402
from src.utils.trading_calendar import sessions_between  # noqa: E402


def _price_coverage(conn: sqlite3.Connection, days: int) -> pd.DataFrame:
    prices = pd.read_sql_query(
        "SELECT ticker, date FROM prices WHERE ticker != 'SPY' ORDER BY ticker, date",
        conn,
    )
    if prices.empty:
        return pd.DataFrame()
    prices["date"] = pd.to_datetime(prices["date"])
    rows = []
    for ticker in TICKERS:
        sub = prices[prices["ticker"] == ticker]
        if sub.empty:
            rows.append({"ticker": ticker, "price_min": None, "price_max": None, "price_rows": 0})
            continue
        rows.append({
            "ticker": ticker,
            "price_min": sub["date"].min().strftime("%Y-%m-%d"),
            "price_max": sub["date"].max().strftime("%Y-%m-%d"),
            "price_rows": len(sub),
        })
    return pd.DataFrame(rows)


def _news_sparse_days(conn: sqlite3.Connection, days: int) -> list[str]:
    news = _load_news_aligned(conn)
    if news.empty:
        return ["  [warn] no cutoff-aligned news in DB"]

    prices = pd.read_sql_query(
        "SELECT DISTINCT date FROM prices WHERE ticker != 'SPY' ORDER BY date",
        conn,
    )
    if prices.empty:
        return ["  [skip] no price dates for gap analysis"]

    prices["date"] = pd.to_datetime(prices["date"]).dt.strftime("%Y-%m-%d")
    recent_dates = sorted(prices["date"].unique())[-days:]
    news["prediction_date"] = news["label_date"]

    lines: list[str] = []
    for ticker in TICKERS:
        ticker_news_dates = set(
            news.loc[news["ticker"] == ticker, "prediction_date"].astype(str)
        )
        zero_days = [d for d in recent_dates if d not in ticker_news_dates]
        if zero_days:
            # Summarize by month
            by_month: dict[str, int] = {}
            for d in zero_days:
                m = d[:7]
                by_month[m] = by_month.get(m, 0) + 1
            month_str = ", ".join(f"{m}:{c}d" for m, c in sorted(by_month.items())[-3:])
            lines.append(
                f"  {ticker:6s}  {len(zero_days):3d} zero-news days in last {days} sessions "
                f"(recent months: {month_str})"
            )
    return lines or ["  OK — every ticker has news on all recent price sessions"]


def _label_lag(conn: sqlite3.Connection) -> list[str]:
    price_max = pd.read_sql_query(
        "SELECT ticker, MAX(date) AS d FROM prices WHERE ticker != 'SPY' GROUP BY ticker",
        conn,
    )
    label_max = pd.read_sql_query(
        "SELECT ticker, MAX(date) AS d FROM labels GROUP BY ticker",
        conn,
    )
    merged = price_max.merge(label_max, on="ticker", suffixes=("_price", "_label"), how="left")
    lines = []
    for _, r in merged.iterrows():
        p = str(r["d_price"])[:10] if pd.notna(r["d_price"]) else "—"
        l = str(r["d_label"])[:10] if pd.notna(r.get("d_label")) else "—"
        lag = "OK"
        if p != "—" and l != "—":
            try:
                from datetime import datetime
                dp = datetime.strptime(p, "%Y-%m-%d").date()
                dl = datetime.strptime(l, "%Y-%m-%d").date()
                n = sessions_between(dl, dp) if dp > dl else 0
                lag = f"{n} sessions behind" if n else "OK"
            except ValueError:
                lag = "?"
        lines.append(f"  {r['ticker']:6s}  prices→{p}  labels→{l}  ({lag})")
    return lines


def _live_row_mismatches() -> list[str]:
    path = PROCESSED_DATA_DIR / "final_ensemble_predictions.csv"
    if not path.exists():
        return ["  [skip] final_ensemble_predictions.csv not found"]
    df = pd.read_csv(path)
    if "has_news" not in df.columns or "n_headlines" not in df.columns:
        return ["  [skip] missing has_news / n_headlines columns"]
    bad = df[(df["has_news"].fillna(0).astype(int) == 0) & (df["n_headlines"].fillna(0) > 0)]
    if bad.empty:
        return ["  OK — no rows with has_news=0 and n_headlines>0"]
    sample = bad[["ticker", "prediction_date", "has_news", "n_headlines"]].head(10)
    lines = [f"  *** {len(bad)} mismatched rows (has_news=0 but headlines exist) ***"]
    for _, r in sample.iterrows():
        lines.append(
            f"    {r['ticker']} {r['prediction_date']}  n_headlines={int(r['n_headlines'])}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data coverage across pipeline")
    parser.add_argument("--days", type=int, default=365, help="Lookback for news gap analysis")
    args = parser.parse_args()

    print("=" * 70)
    print("DATA COVERAGE AUDIT")
    print("=" * 70)

    if not DATABASE_PATH.exists():
        print(f"Database not found: {DATABASE_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DATABASE_PATH))
    try:
        print("\n--- Price coverage ---")
        cov = _price_coverage(conn, args.days)
        if cov.empty:
            print("  No price data.")
        else:
            for _, r in cov.iterrows():
                print(
                    f"  {r['ticker']:6s}  {r['price_min']} → {r['price_max']}  "
                    f"({int(r['price_rows']):,} rows)"
                )

        print(f"\n--- News gaps (last {args.days} price sessions, zero headlines) ---")
        for line in _news_sparse_days(conn, args.days):
            print(line)

        print("\n--- Label lag vs prices ---")
        for line in _label_lag(conn):
            print(line)
    finally:
        conn.close()

    print("\n--- Live row has_news consistency ---")
    for line in _live_row_mismatches():
        print(line)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
