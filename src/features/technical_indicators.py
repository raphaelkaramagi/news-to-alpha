"""Compute technical indicators from price data.

These give the LSTM extra signals beyond raw OHLCV:
- RSI: is the stock overbought (>70) or oversold (<30)?
- MACD: is the trend accelerating or slowing?
- Bollinger Bands: is the price unusually high/low relative to recent history?
- Volume MA: is trading volume above or below normal?

All computed with pandas — no external TA library needed.
"""

import sqlite3
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import DATABASE_PATH

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    def __init__(self, db_path: str | Path = DATABASE_PATH):
        self.db_path = Path(db_path)

    def compute(self, ticker: str) -> pd.DataFrame:
        """
        Pull price data for a ticker and add all indicator columns.
        Returns a DataFrame indexed by date with OHLCV + indicators.
        Early rows will have NaN where there isn't enough history
        for the indicator window (e.g. RSI needs 14 days).
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices "
            "WHERE ticker = ? ORDER BY date ASC",
            conn,
            params=(ticker,),
        )
        conn.close()

        if df.empty:
            logger.warning("No price data for %s", ticker)
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_volume_ma(df)

        return df

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Relative Strength Index (0-100).
        Compares average gains vs losses over the last `period` days.
        """
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=period, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _add_macd(df: pd.DataFrame,
                  fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence.
        macd_histogram > 0 = bullish momentum, < 0 = bearish.
        """
        ema_fast = df["close"].ewm(span=fast, min_periods=fast).mean()
        ema_slow = df["close"].ewm(span=slow, min_periods=slow).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=signal, min_periods=signal).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        return df

    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands: moving average +/- 2 standard deviations.
        bb_position = where price sits in the band (0=lower, 1=upper).
        """
        df["bb_middle"] = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()

        df["bb_upper"] = df["bb_middle"] + (rolling_std * num_std)
        df["bb_lower"] = df["bb_middle"] - (rolling_std * num_std)

        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_width"] = bb_range / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
        return df

    @staticmethod
    def _add_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Volume moving average and ratio.
        volume_ratio > 1 = above-normal trading activity.
        """
        df["volume_ma"] = df["volume"].rolling(window=period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
        return df
