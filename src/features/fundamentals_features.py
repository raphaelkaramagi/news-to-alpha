"""Earnings-proximity, sector, and valuation features from the fundamentals tables.

These are slow-moving / calendar features used by the volatility model and the
walk-forward harness. Earnings proximity is the single biggest free lever for
next-day move size (stocks move much more around earnings).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATABASE_PATH


def load_earnings_dates(db_path: str | Path = DATABASE_PATH) -> dict[str, list[pd.Timestamp]]:
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query("SELECT ticker, earnings_date FROM earnings_dates", conn)
    except Exception:
        return {}
    finally:
        conn.close()
    if df.empty:
        return {}
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    df = df.dropna(subset=["earnings_date"])
    return {tk: sorted(g["earnings_date"].tolist()) for tk, g in df.groupby("ticker")}


def load_fundamentals(db_path: str | Path = DATABASE_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query("SELECT * FROM fundamentals", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def add_earnings_proximity(
    panel: pd.DataFrame,
    *,
    date_col: str = "prediction_date",
    db_path: str | Path = DATABASE_PATH,
    window: int = 2,
) -> pd.DataFrame:
    """Add days_to_earnings (signed) + earnings_window flag to a (ticker, date) panel.

    days_to_earnings: signed calendar days to the NEAREST earnings date
                      (negative = earnings just happened, positive = upcoming).
                      Clipped to +/-30 and 99 when no calendar is available.
    earnings_window : 1 when |days_to_earnings| <= window.
    """
    ed_map = load_earnings_dates(db_path)
    out = panel.copy()
    dates = pd.to_datetime(out[date_col], errors="coerce")

    days = np.full(len(out), 99.0)
    if ed_map:
        for i, (tk, d) in enumerate(zip(out["ticker"].to_numpy(), dates)):
            cal = ed_map.get(tk)
            if not cal or pd.isna(d):
                continue
            diffs = [(e - d).days for e in cal]
            nearest = min(diffs, key=abs)
            days[i] = float(np.clip(nearest, -30, 30))

    out["days_to_earnings"] = days
    out["earnings_window"] = (np.abs(days) <= window).astype(int)
    return out


def add_sector(
    panel: pd.DataFrame,
    *,
    db_path: str | Path = DATABASE_PATH,
) -> pd.DataFrame:
    """Attach a `sector` column (string; 'Unknown' when missing)."""
    fund = load_fundamentals(db_path)
    out = panel.copy()
    if fund.empty or "sector" not in fund.columns:
        out["sector"] = "Unknown"
        return out
    sector_map = dict(zip(fund["ticker"], fund["sector"].fillna("Unknown")))
    out["sector"] = out["ticker"].map(sector_map).fillna("Unknown")
    return out
