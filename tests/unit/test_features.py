"""Tests for technical indicators and sequence generation."""

import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.database.schema import DatabaseSchema
from src.features.technical_indicators import TechnicalIndicators
from src.features.sequence_generator import SequenceGenerator, FEATURE_COLUMNS


@pytest.fixture
def test_db(tmp_path):
    """Create a temp DB with 80 days of fake price data + labels + news.
    Need 80+ days because MACD alone needs ~35 rows before it produces values."""
    db_path = tmp_path / "test.db"
    DatabaseSchema(db_path).create_all_tables()

    conn = sqlite3.connect(db_path)

    np.random.seed(42)
    base_price = 100.0
    dates = pd.bdate_range("2026-01-05", periods=80)
    for i, date in enumerate(dates):
        close = base_price + i * 0.5 + np.random.randn() * 2
        conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("TEST", date.strftime("%Y-%m-%d"), close - 1, close + 1, close - 2, close,
             1000000 + i * 10000),
        )

    rows = conn.execute(
        "SELECT date, close FROM prices WHERE ticker='TEST' ORDER BY date"
    ).fetchall()
    for j in range(len(rows) - 1):
        date_t, close_t = rows[j]
        _, close_t1 = rows[j + 1]
        label = 1 if close_t1 > close_t else 0
        pct = ((close_t1 - close_t) / close_t) * 100
        conn.execute(
            "INSERT INTO labels (ticker, date, label_binary, label_return, close_t, close_t_plus_1) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("TEST", date_t, label, round(pct, 4), close_t, close_t1),
        )

    # Add news articles for the last 10 trading days
    for i in range(70, 80):
        date_str = dates[i].strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO news (ticker, title, url, published_at) VALUES (?, ?, ?, ?)",
            ("TEST", f"Test Corp stock rises on day {i}",
             f"https://example.com/test-{i}",
             f"{date_str}T10:00:00-05:00"),
        )

    conn.commit()
    conn.close()
    return db_path


class TestTechnicalIndicators:
    def test_returns_dataframe_with_all_indicators(self, test_db):
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")

        expected_cols = [
            "daily_return",
            "rsi", "rsi_norm",
            "macd_line", "macd_signal", "macd_histogram",
            "macd_line_rel", "macd_signal_rel", "macd_hist_rel",
            "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
            "volume_ma", "volume_ratio", "volume_ratio_m1",
            "roc_5", "roc_10",
            "atr_14", "atr_rel",
            "obv", "realized_vol_20",
            "market_return", "market_return_5d",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_daily_return_computed(self, test_db):
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")
        dr = df["daily_return"].dropna()
        assert len(dr) > 0
        assert not np.isinf(dr).any()

    def test_rsi_range(self, test_db):
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")
        rsi_valid = df["rsi"].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()

    def test_rsi_norm_range(self, test_db):
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")
        r = df["rsi_norm"].dropna()
        assert (r >= 0).all() and (r <= 1).all()

    def test_atr_positive(self, test_db):
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")
        a = df["atr_14"].dropna()
        assert len(a) > 0
        assert (a >= 0).all()

    def test_empty_ticker(self, test_db):
        ti = TechnicalIndicators(test_db)
        df = ti.compute("DOESNOTEXIST")
        assert df.empty

    def test_feature_column_order(self):
        """FEATURE_COLUMNS is the canonical order the LSTM expects."""
        assert "daily_return" in FEATURE_COLUMNS
        assert "market_return" in FEATURE_COLUMNS
        assert "atr_rel" in FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) >= 20


class TestSequenceGenerator:
    def test_generates_sequences(self, test_db):
        gen = SequenceGenerator(test_db, sequence_length=5)
        X, y, returns, dates = gen.generate("TEST")

        assert len(X) > 0
        assert X.shape[1] == 5
        assert X.shape[2] == len(FEATURE_COLUMNS)
        assert len(y) == len(X)
        assert len(returns) == len(X)
        assert len(dates) == len(X)

    def test_labels_are_binary(self, test_db):
        gen = SequenceGenerator(test_db, sequence_length=5)
        _, y, _, _ = gen.generate("TEST")
        assert set(y).issubset({0, 1})

    def test_sequences_are_raw_unscaled(self, test_db):
        """Unlike the old generator, values are raw - not per-window normalized."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        X, _, _, _ = gen.generate("TEST")
        # Prices in the fixture grow to ~140 - raw values should exceed 1.
        assert X.max() > 1.0

    def test_dates_are_strings(self, test_db):
        gen = SequenceGenerator(test_db, sequence_length=5)
        _, _, _, dates = gen.generate("TEST")
        for d in dates:
            assert len(d) == 10 and d[4] == "-" and d[7] == "-"

    def test_no_labels_returns_empty(self, tmp_path):
        db_path = tmp_path / "nolabel.db"
        DatabaseSchema(db_path).create_all_tables()

        conn = sqlite3.connect(db_path)
        for i in range(10):
            conn.execute(
                "INSERT INTO prices (ticker, date, close, volume) VALUES (?, ?, ?, ?)",
                ("X", f"2026-01-{i+1:02d}", 100 + i, 1000000),
            )
        conn.commit()
        conn.close()

        gen = SequenceGenerator(db_path, sequence_length=5)
        X, y, returns, dates = gen.generate("X")
        assert len(X) == 0


