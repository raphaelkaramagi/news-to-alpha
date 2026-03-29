"""Tests for technical indicators, sequence generation, and text features."""

import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.database.schema import DatabaseSchema
from src.features.technical_indicators import TechnicalIndicators
from src.features.sequence_generator import SequenceGenerator, FEATURE_COLUMNS
from src.features.text_features import TextFeatureExtractor


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
        """compute() should produce all 17 feature columns."""
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")

        expected_cols = [
            "daily_return",
            "rsi", "macd_line", "macd_signal", "macd_histogram",
            "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
            "volume_ma", "volume_ratio",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_daily_return_computed(self, test_db):
        """daily_return should be the percentage change of close."""
        ti = TechnicalIndicators(test_db)
        df = ti.compute("TEST")
        dr = df["daily_return"].dropna()
        assert len(dr) > 0
        assert not np.isinf(dr).any()

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

    def test_feature_column_count(self, test_db):
        """FEATURE_COLUMNS list should have 17 entries (OHLCV + daily_return + indicators)."""
        assert len(FEATURE_COLUMNS) == 17
        assert "daily_return" in FEATURE_COLUMNS


class TestSequenceGenerator:
    def test_generates_sequences(self, test_db):
        """Should produce sequences with correct shape (17 features)."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        X, y, dates = gen.generate("TEST")

        assert len(X) > 0
        assert X.shape[1] == 5
        assert X.shape[2] == len(FEATURE_COLUMNS)
        assert len(y) == len(X)
        assert len(dates) == len(X)

    def test_labels_are_binary(self, test_db):
        """All labels should be 0 or 1."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        _, y, _ = gen.generate("TEST")
        assert set(y).issubset({0, 1})

    def test_normalization_bounds(self, test_db):
        """Features should be in [0, 1] after per-window normalization."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        X, _, _ = gen.generate("TEST")

        assert X.min() >= -0.001
        assert X.max() <= 1.001

    def test_constant_column_normalized_to_half(self, test_db):
        """If a feature is constant across a window, it should become 0.5."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        X, _, _ = gen.generate("TEST")

        for sample_idx in range(min(5, len(X))):
            window = X[sample_idx]
            for col_idx in range(window.shape[1]):
                col = window[:, col_idx]
                if col.std() < 1e-9:
                    assert abs(col[0] - 0.5) < 0.01, \
                        f"Constant column should be 0.5, got {col[0]}"

    def test_dates_are_strings(self, test_db):
        """Returned dates should be YYYY-MM-DD strings."""
        gen = SequenceGenerator(test_db, sequence_length=5)
        _, _, dates = gen.generate("TEST")
        for d in dates:
            assert len(d) == 10 and d[4] == "-" and d[7] == "-"

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


class TestTextFeatureExtractor:
    def test_prepare_returns_aligned_data(self, test_db):
        """prepare() should return texts, labels, and metadata of equal length."""
        ext = TextFeatureExtractor(db_path=test_db)
        texts, labels, metadata = ext.prepare(["TEST"])

        assert len(texts) > 0
        assert len(texts) == len(labels) == len(metadata)

    def test_metadata_is_ticker_date_pairs(self, test_db):
        """Each metadata entry should be (ticker, date_string)."""
        ext = TextFeatureExtractor(db_path=test_db)
        _, _, metadata = ext.prepare(["TEST"])

        for ticker, date_str in metadata:
            assert ticker == "TEST"
            assert len(date_str) == 10

    def test_labels_are_binary(self, test_db):
        """Labels from prepare() should only be 0 or 1."""
        ext = TextFeatureExtractor(db_path=test_db)
        _, labels, _ = ext.prepare(["TEST"])
        assert set(labels).issubset({0, 1})

    def test_fit_transform_produces_sparse_matrix(self, test_db):
        """fit_transform should return a sparse matrix with correct row count."""
        ext = TextFeatureExtractor(db_path=test_db, max_features=50)
        texts, _, _ = ext.prepare(["TEST"])

        X = ext.fit_transform(texts)
        assert X.shape[0] == len(texts)
        assert X.shape[1] <= 50

    def test_transform_after_fit(self, test_db):
        """transform() should work after fit() and produce same column count."""
        ext = TextFeatureExtractor(db_path=test_db, max_features=50)
        texts, _, _ = ext.prepare(["TEST"])

        ext.fit(texts)
        X = ext.transform(texts)
        assert X.shape[0] == len(texts)

    def test_transform_before_fit_raises(self, test_db):
        """transform() before fit() should raise RuntimeError."""
        ext = TextFeatureExtractor(db_path=test_db)
        with pytest.raises(RuntimeError):
            ext.transform(["some text"])

    def test_no_news_returns_empty(self, tmp_path):
        """Ticker with labels but no news should return empty lists."""
        db_path = tmp_path / "nonews.db"
        DatabaseSchema(db_path).create_all_tables()

        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO prices (ticker, date, close) VALUES ('Z', '2026-01-05', 100)")
        conn.execute(
            "INSERT INTO labels (ticker, date, label_binary, label_return, close_t, close_t_plus_1) "
            "VALUES ('Z', '2026-01-05', 1, 1.0, 100, 101)")
        conn.commit()
        conn.close()

        ext = TextFeatureExtractor(db_path=db_path)
        texts, labels, meta = ext.prepare(["Z"])
        assert len(texts) == 0
