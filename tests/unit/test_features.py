"""Tests for technical indicators and sequence generation."""

import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.database.schema import DatabaseSchema
from src.features.technical_indicators import TechnicalIndicators
from src.features.sequence_generator import SequenceGenerator


@pytest.fixture
def test_db(tmp_path):
    """Create a temp DB with 80 days of fake price data + labels.
    Need 80+ days because MACD alone needs ~35 rows before it produces values."""
    db_path = tmp_path / "test.db"
    DatabaseSchema(db_path).create_all_tables()

    conn = sqlite3.connect(db_path)

    # 80 trading days of slightly noisy upward-trending prices
    np.random.seed(42)
    base_price = 100.0
    dates = pd.bdate_range("2026-01-05", periods=80)  # business days only
    for i, date in enumerate(dates):
        close = base_price + i * 0.5 + np.random.randn() * 2
        conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("TEST", date.strftime("%Y-%m-%d"), close - 1, close + 1, close - 2, close,
             1000000 + i * 10000),
        )

    # Generate labels for consecutive pairs
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

    conn.commit()
    conn.close()
    return db_path


class TestTechnicalIndicators:
    def test_returns_dataframe_with_indicators(self, test_db):
        """compute() should add RSI, MACD, and Bollinger columns."""
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")

        expected_cols = ["rsi", "macd_line", "macd_signal", "macd_histogram",
                         "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
                         "volume_ma", "volume_ratio"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_rsi_range(self, test_db):
        """RSI should be between 0 and 100 (where not NaN)."""
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")
        rsi_valid = df["rsi"].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()

    def test_empty_ticker(self, test_db):
        """Ticker with no data should return empty DataFrame."""
        ti = TechnicalIndicators(test_db)
        df = ti.compute("DOESNOTEXIST")
        assert df.empty


class TestSequenceGenerator:
    def test_generates_sequences(self, test_db):
        """Should produce sequences with correct shape."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        X, y, dates = gen.generate("TEST")

        assert len(X) > 0
        assert X.shape[1] == 5   # sequence length
        assert X.shape[2] == 16  # number of features
        assert len(y) == len(X)
        assert len(dates) == len(X)

    def test_labels_are_binary(self, test_db):
        """All labels should be 0 or 1."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        _, y, _ = gen.generate("TEST")
        assert set(y).issubset({0, 1})

    def test_normalization(self, test_db):
        """Each feature in each window should be in [0, 1] after normalization."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        X, _, _ = gen.generate("TEST")

        assert X.min() >= -0.001  # small float tolerance
        assert X.max() <= 1.001

    def test_no_labels_returns_empty(self, tmp_path):
        """Ticker with prices but no labels should return empty."""
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
        X, y, dates = gen.generate("X")
        assert len(X) == 0
