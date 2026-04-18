"""Compute technical indicators + market-wide regime features from price data.

The LSTM sees every ticker's 60-day window through the same feature lens:

Price-derived signals (per ticker):
- daily_return  - pct change of close day-over-day
- rsi_norm      - RSI / 100 (bounded 0-1)
- macd_line, macd_signal, macd_histogram + their /close ratios
- bb_middle/upper/lower/width/position
- volume_ma, volume_ratio, volume_ratio_m1 (= ratio - 1, centered on zero)
- roc_5, roc_10 - rate-of-change (pct) over 5 and 10 days
- atr_14, atr_rel (ATR / close) - realized intra-day range
- obv            - on-balance volume
- realized_vol_20 - 20-day annualized std of log returns
- volume_zscore_20 - 20-day rolling z-score of raw volume
- excess_return  - daily_return - market_return (idiosyncratic move)

Market-wide regime (same value for every ticker on a given date):
- market_return        - SPY daily_return
- market_return_5d     - SPY 5-day ROC

All heavy lifting is pandas; no external TA library.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATABASE_PATH

logger = logging.getLogger(__name__)

MARKET_TICKER = "SPY"


class TechnicalIndicators:
    def __init__(self, db_path: str | Path = DATABASE_PATH):
        self.db_path = Path(db_path)
        self._market_cache: pd.DataFrame | None = None

    def compute(self, ticker: str) -> pd.DataFrame:
        """Pull price data for a ticker and add all indicator columns.

        Returns a DataFrame indexed by date with OHLCV + indicators + market
        features.  Early rows are NaN where the indicator window hasn't filled
        (e.g. RSI needs 14 days).
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

        df = self._add_daily_return(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_volume_ma(df)
        df = self._add_roc(df)
        df = self._add_atr(df)
        df = self._add_obv(df)
        df = self._add_realized_vol(df)
        df = self._add_volume_zscore(df)
        df = self._merge_market_features(df)
        df = self._add_excess_return(df)
        df = self._add_derived_ratios(df)

        return df

    @staticmethod
    def _add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
        """Percentage change in close price day-over-day (% units)."""
        df["daily_return"] = df["close"].pct_change() * 100
        return df

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index (0-100)."""
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=period, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_norm"] = df["rsi"] / 100.0
        return df

    @staticmethod
    def _add_macd(df: pd.DataFrame,
                  fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Moving Average Convergence Divergence."""
        ema_fast = df["close"].ewm(span=fast, min_periods=fast).mean()
        ema_slow = df["close"].ewm(span=slow, min_periods=slow).mean()

        df["macd_line"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd_line"].ewm(span=signal, min_periods=signal).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        return df

    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame, period: int = 20,
                             num_std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands and position inside them (0 = lower band, 1 = upper)."""
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
        """Volume moving average and ratio (ratio > 1 = above-normal activity)."""
        df["volume_ma"] = df["volume"].rolling(window=period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
        df["volume_ratio_m1"] = df["volume_ratio"] - 1.0
        return df

    @staticmethod
    def _add_roc(df: pd.DataFrame) -> pd.DataFrame:
        """Rate of Change over 5 and 10 days (in % units)."""
        df["roc_5"] = df["close"].pct_change(5) * 100
        df["roc_10"] = df["close"].pct_change(10) * 100
        return df

    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range over `period` days; atr_rel = atr / close."""
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(span=period, min_periods=period).mean()
        df["atr_rel"] = df["atr_14"] / df["close"].replace(0, np.nan)
        return df

    @staticmethod
    def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume (cumulative signed volume)."""
        direction = np.sign(df["close"].diff().fillna(0)).astype(float)
        df["obv"] = (direction * df["volume"]).cumsum()
        return df

    @staticmethod
    def _add_realized_vol(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Rolling realized volatility of log returns (annualized, % units)."""
        log_ret = np.log(df["close"] / df["close"].shift(1))
        df["realized_vol_20"] = (
            log_ret.rolling(window=period).std() * np.sqrt(252) * 100
        )
        return df

    @staticmethod
    def _add_volume_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """20-day rolling z-score of raw volume (centered, unitless).

        More informative than volume_ratio when variance is large: catches
        genuine volume shocks while dampening small fluctuations.
        """
        mean = df["volume"].rolling(window=period).mean()
        std = df["volume"].rolling(window=period).std().replace(0, np.nan)
        df["volume_zscore_20"] = (df["volume"] - mean) / std
        return df

    @staticmethod
    def _add_excess_return(df: pd.DataFrame) -> pd.DataFrame:
        """daily_return minus SPY's daily_return - idiosyncratic move (% units)."""
        if "market_return" not in df.columns:
            df["excess_return"] = df.get("daily_return", 0.0)
            return df
        df["excess_return"] = (
            df["daily_return"].fillna(0.0) - df["market_return"].fillna(0.0)
        )
        return df

    def _merge_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Left-join SPY's daily_return and 5-day ROC onto the ticker's date index."""
        market = self._load_market()
        if market is None or market.empty:
            df["market_return"] = 0.0
            df["market_return_5d"] = 0.0
            return df
        df = df.join(market, how="left")
        df["market_return"] = df["market_return"].fillna(0.0)
        df["market_return_5d"] = df["market_return_5d"].fillna(0.0)
        return df

    def _load_market(self) -> pd.DataFrame | None:
        """Pull SPY prices once per instance and compute its return series."""
        if self._market_cache is not None:
            return self._market_cache

        conn = sqlite3.connect(self.db_path)
        spy = pd.read_sql_query(
            "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date ASC",
            conn,
            params=(MARKET_TICKER,),
        )
        conn.close()
        if spy.empty:
            logger.info("%s prices not in DB - market regime features will be 0.",
                        MARKET_TICKER)
            self._market_cache = pd.DataFrame()
            return self._market_cache

        spy["date"] = pd.to_datetime(spy["date"])
        spy = spy.set_index("date").sort_index()
        spy["market_return"] = spy["close"].pct_change() * 100
        spy["market_return_5d"] = spy["close"].pct_change(5) * 100
        self._market_cache = spy[["market_return", "market_return_5d"]]
        return self._market_cache

    @staticmethod
    def _add_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """Express scale-dependent indicators as ratios to current close."""
        close = df["close"].replace(0, np.nan)
        df["macd_line_rel"] = df["macd_line"] / close
        df["macd_signal_rel"] = df["macd_signal"] / close
        df["macd_hist_rel"] = df["macd_histogram"] / close
        return df
