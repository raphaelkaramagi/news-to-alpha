"""Generate 60-day sequences for LSTM training (no per-window normalization).

The old per-window min-max destroyed absolute scale across windows (RSI=70
could become 0.0 in one window and 1.0 in another).  Scaling now happens
once, globally, in `scripts/train_lstm.py` using a StandardScaler fit on
the training split; level features get scaled, scale-invariant features
(returns, ratios, oscillators) go through unchanged.

This module is concerned only with:
    - pulling OHLCV + indicators + market regime for a ticker
    - dropping NaN warmup rows
    - emitting sliding windows (X, y, dates)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATABASE_PATH, LSTM_CONFIG
from src.features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

SCALE_INVARIANT_FEATURES: list[str] = [
    "daily_return",
    "rsi_norm",
    "macd_hist_rel",
    "macd_line_rel",
    "macd_signal_rel",
    "bb_position",
    "bb_width",
    "volume_ratio_m1",
    "roc_5",
    "roc_10",
    "atr_rel",
    "realized_vol_20",
    "market_return",
    "market_return_5d",
]

LEVEL_FEATURES: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "macd_line",
    "macd_signal",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "volume_ma",
    "obv",
]

FEATURE_COLUMNS: list[str] = SCALE_INVARIANT_FEATURES + LEVEL_FEATURES

SCALE_INVARIANT_IDX: list[int] = list(range(len(SCALE_INVARIANT_FEATURES)))
LEVEL_IDX: list[int] = list(
    range(len(SCALE_INVARIANT_FEATURES), len(FEATURE_COLUMNS))
)


def _label_columns(horizon: int) -> tuple[str, str]:
    """Return (label_col, return_col) for the requested horizon."""
    if horizon == 1:
        return "label_binary", "label_return"
    if horizon == 3:
        return "label_binary_h3", "return_h3"
    raise ValueError(f"Unsupported horizon {horizon}; expected 1 or 3")


class SequenceGenerator:
    def __init__(self, db_path: str | Path = DATABASE_PATH,
                 sequence_length: int | None = None,
                 horizon: int = 1):
        self.db_path = Path(db_path)
        self.seq_len = sequence_length or LSTM_CONFIG["sequence_length"]
        self.horizon = horizon
        self._label_col, self._return_col = _label_columns(horizon)

    def generate(
        self, ticker: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Build raw (unscaled) sequences for one ticker.

        Returns:
            X       : (num_samples, seq_len, num_features)  float32
            y       : (num_samples,)  int32 (0 = down, 1 = up)
            returns : (num_samples,)  float32 pct return of the target horizon
            dates   : list[str]  label dates (YYYY-MM-DD), one per sample
        """
        ti = TechnicalIndicators(self.db_path)
        df = ti.compute(ticker)

        if df.empty:
            return np.array([]), np.array([]), np.array([]), []

        conn = sqlite3.connect(self.db_path)
        labels_df = pd.read_sql_query(
            f"SELECT date, {self._label_col} AS label, {self._return_col} AS ret "
            "FROM labels WHERE ticker = ? ORDER BY date ASC",
            conn,
            params=(ticker,),
        )
        conn.close()

        if labels_df.empty:
            logger.warning("%s: no labels found - run generate_labels.py first", ticker)
            return np.array([]), np.array([]), np.array([]), []

        labels_df = labels_df.dropna(subset=["label"])
        if labels_df.empty:
            logger.warning(
                "%s: no non-null labels for horizon=%d - regenerate labels",
                ticker, self.horizon,
            )
            return np.array([]), np.array([]), np.array([]), []

        labels_df["date"] = pd.to_datetime(labels_df["date"])
        labels_df = labels_df.set_index("date")

        indicator_df = df[FEATURE_COLUMNS].dropna()
        if len(indicator_df) < self.seq_len + 1:
            logger.warning(
                "%s: only %d valid rows, need at least %d for one sequence",
                ticker, len(indicator_df), self.seq_len + 1,
            )
            return np.array([]), np.array([]), np.array([]), []

        values = indicator_df.to_numpy(dtype=np.float32)
        dates_index = indicator_df.index

        X_list, y_list, ret_list, date_list = [], [], [], []
        for i in range(self.seq_len, len(indicator_df)):
            window = values[i - self.seq_len : i]
            label_date = dates_index[i - 1]
            if label_date not in labels_df.index:
                continue
            row = labels_df.loc[label_date]
            X_list.append(window)
            y_list.append(int(row["label"]))
            ret_list.append(float(row["ret"]) if pd.notna(row["ret"]) else 0.0)
            date_list.append(label_date.strftime("%Y-%m-%d"))

        if not X_list:
            logger.warning("%s: no sequences could be built", ticker)
            return np.array([]), np.array([]), np.array([]), []

        X = np.stack(X_list).astype(np.float32)
        y = np.asarray(y_list, dtype=np.int32)
        returns = np.asarray(ret_list, dtype=np.float32)

        logger.info("%s: %d sequences (shape %s), %.0f%% up (h=%d)",
                    ticker, len(X), X.shape, y.mean() * 100, self.horizon)
        return X, y, returns, date_list
