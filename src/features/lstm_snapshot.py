"""Readable LSTM input snapshot for one (ticker, prediction_date)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATABASE_PATH
from src.features.technical_indicators import TechnicalIndicators

# Subset of sequence features shown in the UI (ensemble does NOT see these directly).
SNAPSHOT_FIELDS: list[tuple[str, str, str]] = [
    ("daily_return", "Prior day return", "%"),
    ("rsi_norm", "RSI (normalized)", "0–1"),
    ("market_return", "SPY day return", "%"),
    ("market_return_5d", "SPY 5-day move", "%"),
    ("vix_level", "VIX level", ""),
    ("vix_change", "VIX day change", "%"),
    ("realized_vol_20", "20-day volatility", "%"),
    ("bb_position", "Bollinger position", "0–1"),
    ("volume_ratio_m1", "Volume vs 20d avg", "ratio"),
    ("excess_return", "Return vs SPY", "%"),
]


def get_lstm_snapshot(
    ticker: str,
    prediction_date: str,
    db_path: str | Path = DATABASE_PATH,
) -> dict[str, object]:
    """Last-day technical/regime values for the LSTM window ending on ``prediction_date``."""
    ti = TechnicalIndicators(db_path)
    df = ti.compute(ticker.upper())
    if df.empty:
        return {"available": False, "fields": []}

    target = pd.Timestamp(prediction_date)
    if target not in df.index:
        # Allow string match on normalized index
        matches = df.index[df.index.strftime("%Y-%m-%d") == prediction_date]
        if len(matches) == 0:
            return {"available": False, "fields": []}
        row = df.loc[matches[-1]]
    else:
        row = df.loc[target]

    fields: list[dict[str, object]] = []
    for key, label, unit in SNAPSHOT_FIELDS:
        if key not in row.index or pd.isna(row[key]):
            continue
        val = float(row[key])
        fields.append({"key": key, "label": label, "value": val, "unit": unit})

    return {
        "available": len(fields) > 0,
        "note": (
            "These feed the price (LSTM) model only. The ensemble combiner sees "
            "price P(UP), headline scores, SPY 5d, headline count, and agreement flags — not raw RSI/VIX."
        ),
        "fields": fields,
    }
