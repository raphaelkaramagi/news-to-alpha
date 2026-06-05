#!/usr/bin/env python3
"""Collect free macro/regime series from FRED (no API key required).

Uses the public keyless CSV endpoint:
    https://fred.stlouisfed.orgx/graph/fredgraph.csv?id=<SERIES>

Default series (daily, regime-relevant):
    DGS2            - 2-year Treasury yield
    DGS10           - 10-year Treasury yield
    T10Y2Y          - 10y-2y slope (recession/regime signal)
    BAMLH0A0HYM2    - high-yield credit spread (risk appetite)
    VIXCLS          - CBOE VIX close (already have VIX, kept for cross-check)

Stored long-format in the `macro` table (date, series_id, value).

Usage
-----
  python scripts/collect_macro.py
  python scripts/collect_macro.py --series DGS10 T10Y2Y
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402

DEFAULT_SERIES = ["DGS2", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "VIXCLS"]
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def _fetch_series(sid: str, retries: int = 3) -> list[tuple[str, float]]:
    import time

    import pandas as pd
    import requests

    url = FRED_CSV.format(sid=sid)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "news-to-alpha/1.0"})
            resp.raise_for_status()
            text = resp.text
            break
        except Exception as exc:  # network / timeout — back off and retry
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise last_exc  # type: ignore[misc]
    df = pd.read_csv(io.StringIO(text))
    # FRED CSV: first col is the date (named DATE or observation_date), second is the series.
    date_col = df.columns[0]
    val_col = df.columns[-1]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    out = []
    for _, r in df.iterrows():
        out.append((str(r[date_col])[:10], float(r[val_col])))
    return out


def collect(series: list[str], db_path: Path = DATABASE_PATH) -> dict:
    DatabaseSchema(db_path).create_all_tables()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    stats = {"rows": 0, "failed": []}
    for sid in series:
        try:
            rows = _fetch_series(sid)
            cur.executemany(
                "INSERT OR REPLACE INTO macro (date, series_id, value) VALUES (?, ?, ?)",
                [(d, sid, v) for d, v in rows],
            )
            stats["rows"] += len(rows)
            print(f"  {sid}: {len(rows)} observations")
        except Exception as exc:
            print(f"  {sid}: FAILED — {exc}")
            stats["failed"].append(sid)
    conn.commit()
    conn.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect free FRED macro series (keyless)")
    ap.add_argument("--series", nargs="*", default=None)
    args = ap.parse_args()
    series = args.series or DEFAULT_SERIES
    print(f"Collecting {len(series)} FRED series ...")
    stats = collect(series)
    print(f"\nDone. rows={stats['rows']}  failed={stats['failed']}")


if __name__ == "__main__":
    main()
