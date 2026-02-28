"""Generate 60-day sequences for LSTM training.

An LSTM needs fixed-length input windows. We slide a 60-day window across each
ticker's price+indicator data and pair each window with the next day's label.

Example: if we have 100 days of data, we get 40 training samples
(days 1-60 -> label for day 60, days 2-61 -> label for day 61, etc.)

Features are normalized per-window to help the LSTM train. Raw price values
vary wildly between stocks ($3 for PLTR vs $250 for AAPL), so we scale each
feature to 0-1 within the window.
"""

import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATABASE_PATH, LSTM_CONFIG
from src.features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# Which columns from the indicator DataFrame become LSTM features
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "rsi", "macd_line", "macd_signal", "macd_histogram",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
    "volume_ma", "volume_ratio",
]


class SequenceGenerator:
    def __init__(self, db_path: str | Path = DATABASE_PATH,
                 sequence_length: int | None = None):
        self.db_path = Path(db_path)
        # Default 60 from config, but can override for testing
        self.seq_len = sequence_length or LSTM_CONFIG["sequence_length"]

    def generate(self, ticker: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build sequences for one ticker.

        Returns:
            X: array of shape (num_samples, seq_len, num_features)
            y: array of shape (num_samples,) — 1=up, 0=down
            dates: list of prediction dates (the date each label corresponds to)
        """
        # Step 1: get price data with indicators
        ti = TechnicalIndicators(self.db_path)
        df = ti.compute(ticker)

        if df.empty:
            return np.array([]), np.array([]), []

        # Step 2: get labels
        conn = sqlite3.connect(self.db_path)
        labels_df = pd.read_sql_query(
            "SELECT date, label_binary FROM labels WHERE ticker = ? ORDER BY date ASC",
            conn,
            params=(ticker,),
        )
        conn.close()

        if labels_df.empty:
            logger.warning("%s: no labels found — run generate_labels.py first", ticker)
            return np.array([]), np.array([]), []

        labels_df["date"] = pd.to_datetime(labels_df["date"])
        labels_df = labels_df.set_index("date")

        # Step 3: drop rows where any indicator is NaN (early rows without enough history)
        indicator_df = df[FEATURE_COLUMNS].dropna()

        if len(indicator_df) < self.seq_len + 1:
            logger.warning("%s: only %d valid rows, need at least %d for one sequence",
                           ticker, len(indicator_df), self.seq_len + 1)
            return np.array([]), np.array([]), []

        # Step 4: build sliding windows
        X_list = []
        y_list = []
        date_list = []

        values = indicator_df.values  # numpy array for speed
        dates_index = indicator_df.index

        for i in range(self.seq_len, len(indicator_df)):
            # The window is the previous seq_len days
            window = values[i - self.seq_len : i]

            # The label date is the last day of the window
            label_date = dates_index[i - 1]

            if label_date not in labels_df.index:
                continue

            # Normalize each feature to 0-1 within this window
            window_normalized = self._normalize_window(window)

            X_list.append(window_normalized)
            y_list.append(labels_df.loc[label_date, "label_binary"])
            date_list.append(str(label_date.date()))

        if not X_list:
            logger.warning("%s: no sequences could be built", ticker)
            return np.array([]), np.array([]), []

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        logger.info("%s: %d sequences (shape %s), %.0f%% up",
                     ticker, len(X), X.shape, y.mean() * 100)
        return X, y, date_list

    @staticmethod
    def _normalize_window(window: np.ndarray) -> np.ndarray:
        """
        Min-max normalize each feature column to [0, 1] within this window.
        If a column is constant (max == min), set it to 0.5.
        """
        mins = window.min(axis=0)
        maxs = window.max(axis=0)
        ranges = maxs - mins

        # Avoid division by zero for constant columns
        ranges[ranges == 0] = 1.0

        return (window - mins) / ranges
