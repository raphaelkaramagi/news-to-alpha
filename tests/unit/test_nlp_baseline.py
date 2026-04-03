"""Tests for the NLP baseline model (TF-IDF + logistic regression)."""

import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.database.schema import DatabaseSchema
from src.models.nlp_baseline import NLPBaseline


@pytest.fixture
def nlp_db(tmp_path):
    """Database with labels and enough news for TF-IDF to work.

    Creates 30 days of prices, labels, and 2 articles per day for 2 tickers,
    giving ~120 total samples — enough for a meaningful train/test split.
    """
    db_path = tmp_path / "nlp_test.db"
    DatabaseSchema(db_path).create_all_tables()

    conn = sqlite3.connect(db_path)
    np.random.seed(42)

    tickers = ["AA", "BB"]
    dates = pd.bdate_range("2026-01-05", periods=30)

    headlines = [
        "Stock surges on strong earnings beat",
        "Company reports disappointing quarterly revenue",
        "Analyst upgrades to strong buy rating",
        "Shares drop after product recall announced",
        "New partnership deal boosts market confidence",
        "Revenue misses expectations by wide margin",
    ]

    for ticker in tickers:
        base = 100.0
        for i, date in enumerate(dates):
            close = base + np.random.randn() * 5
            conn.execute(
                "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ticker, date.strftime("%Y-%m-%d"),
                 close - 1, close + 1, close - 2, close, 1000000),
            )

            if i < len(dates) - 1:
                next_close = base + np.random.randn() * 5
                label = 1 if next_close > close else 0
                pct = ((next_close - close) / close) * 100
                conn.execute(
                    "INSERT INTO labels (ticker, date, label_binary, label_return, "
                    "close_t, close_t_plus_1) VALUES (?, ?, ?, ?, ?, ?)",
                    (ticker, date.strftime("%Y-%m-%d"), label, round(pct, 4),
                     close, next_close),
                )
                base = next_close
            else:
                base = close

            # 2 news articles per day
            for j in range(2):
                h = headlines[(i + j) % len(headlines)]
                conn.execute(
                    "INSERT INTO news (ticker, title, url, published_at) "
                    "VALUES (?, ?, ?, ?)",
                    (ticker, f"{ticker} {h}",
                     f"https://example.com/{ticker}-{i}-{j}",
                     f"{date.strftime('%Y-%m-%d')}T10:00:00-05:00"),
                )

    conn.commit()
    conn.close()
    return db_path


class TestNLPBaseline:
    def test_prepare_and_train(self, nlp_db):
        """Full pipeline: prepare data, fit vectorizer, train classifier."""
        model = NLPBaseline(db_path=nlp_db, max_features=50)
        texts, labels, meta = model.extractor.prepare(["AA", "BB"])

        assert len(texts) > 0
        X = model.extractor.fit_transform(texts)
        model.train(X, labels)

        preds = model.predict(X)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(labels)

    def test_predict_proba_range(self, nlp_db):
        """predict_proba should return values in [0, 1]."""
        model = NLPBaseline(db_path=nlp_db, max_features=50)
        texts, labels, _ = model.extractor.prepare(["AA", "BB"])
        X = model.extractor.fit_transform(texts)
        model.train(X, labels)

        probs = model.predict_proba(X)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_evaluate_returns_accuracy(self, nlp_db):
        """evaluate() should return a float accuracy in [0, 1]."""
        model = NLPBaseline(db_path=nlp_db, max_features=50)
        texts, labels, _ = model.extractor.prepare(["AA", "BB"])
        X = model.extractor.fit_transform(texts)
        model.train(X, labels)

        acc = model.evaluate(X, labels, split_name="Train")
        assert 0.0 <= acc <= 1.0

    def test_save_and_load(self, nlp_db, tmp_path):
        """Saved model should reproduce predictions after loading."""
        model = NLPBaseline(db_path=nlp_db, max_features=50)
        texts, labels, _ = model.extractor.prepare(["AA", "BB"])
        X = model.extractor.fit_transform(texts)
        model.train(X, labels)

        save_path = tmp_path / "nlp_model.joblib"
        model.save(save_path)

        loaded = NLPBaseline.load(save_path, db_path=nlp_db)
        X_loaded = loaded.extractor.transform(texts)

        np.testing.assert_array_equal(
            model.predict(X), loaded.predict(X_loaded)
        )

    def test_balanced_class_weights(self, nlp_db):
        """Classifier should use balanced class weights."""
        model = NLPBaseline(db_path=nlp_db)
        assert model.classifier.class_weight == "balanced"

    def test_no_data_returns_empty(self, tmp_path):
        """Ticker with no news/labels should return empty arrays."""
        db_path = tmp_path / "empty.db"
        DatabaseSchema(db_path).create_all_tables()

        model = NLPBaseline(db_path=db_path)
        texts, labels, meta = model.extractor.prepare(["ZZZ"])
        assert len(texts) == 0
        assert len(labels) == 0
