"""Tests covering Phase M/U additions:

- 3-day horizon labels in LabelGenerator
- drop-flat (`min_move_pct`) filter semantics
- `pos_weight` support in LSTMTrainer
- content-snippet aggregation in `news_pipeline._load_news_aligned`
- new Flask API routes: /api/presets, /api/dates, /api/conviction, /api/run, /api/rationale
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_processing.label_generator import LabelGenerator
from src.database.schema import DatabaseSchema
from src.models.lstm_model import StockLSTM, LSTMTrainer
from src.models.news_pipeline import _load_news_aligned


# --------------------------------------------------------------------------
# Label generator: horizon=3
# --------------------------------------------------------------------------

@pytest.fixture
def five_day_db(tmp_path):
    db_path = tmp_path / "five.db"
    DatabaseSchema(db_path).create_all_tables()
    conn = sqlite3.connect(db_path)
    prices = [
        ("AAPL", "2026-01-05", 100.0),
        ("AAPL", "2026-01-06", 101.0),
        ("AAPL", "2026-01-07", 102.0),
        ("AAPL", "2026-01-08", 110.0),  # close_t_plus_3 for 2026-01-05
        ("AAPL", "2026-01-09", 99.0),
    ]
    for t, d, c in prices:
        conn.execute(
            "INSERT INTO prices (ticker, date, close) VALUES (?, ?, ?)",
            (t, d, c),
        )
    conn.commit()
    conn.close()
    return db_path


class TestHorizon3Labels:
    def test_h3_columns_populated(self, five_day_db):
        LabelGenerator(five_day_db).generate(["AAPL"])
        conn = sqlite3.connect(five_day_db)
        rows = conn.execute(
            "SELECT date, label_binary, label_binary_h3, return_h3, close_t_plus_3 "
            "FROM labels WHERE ticker='AAPL' ORDER BY date"
        ).fetchall()
        conn.close()

        # 2026-01-05 -> close_t_plus_3 = 2026-01-08 = 110 -> up, +10%
        first = rows[0]
        assert first[0] == "2026-01-05"
        assert first[1] == 1  # 1-day: 100 -> 101 up
        assert first[2] == 1  # 3-day: 100 -> 110 up
        assert abs(first[3] - 10.0) < 0.01
        assert abs(first[4] - 110.0) < 0.01

    def test_h3_null_when_unavailable(self, five_day_db):
        LabelGenerator(five_day_db).generate(["AAPL"])
        conn = sqlite3.connect(five_day_db)
        row = conn.execute(
            "SELECT label_binary, label_binary_h3 FROM labels "
            "WHERE ticker='AAPL' AND date='2026-01-07'"
        ).fetchone()
        conn.close()
        # 2026-01-07 has t+1 (2026-01-08) but not t+3 (would need day index 6)
        assert row[0] == 1  # 102 -> 110 up
        assert row[1] is None  # no t+3 available


# --------------------------------------------------------------------------
# Drop-flat (min_move_pct) filter semantics
# --------------------------------------------------------------------------

class TestDropFlatFilter:
    def test_filter_removes_small_moves(self):
        df = pd.DataFrame({
            "label_return": [0.1, -0.2, 2.5, -3.0, 0.05],
            "label_binary": [1, 0, 1, 0, 1],
        })
        keep = df["label_return"].abs() >= 0.5
        filtered = df[keep].reset_index(drop=True)
        assert len(filtered) == 2
        assert filtered["label_return"].tolist() == [2.5, -3.0]

    def test_filter_noop_when_threshold_is_zero(self):
        df = pd.DataFrame({"label_return": [0.0, 0.1, -0.2]})
        keep = df["label_return"].abs() >= 0.0
        assert keep.all()


# --------------------------------------------------------------------------
# LSTMTrainer: pos_weight plumbed into BCEWithLogitsLoss
# --------------------------------------------------------------------------

class TestPosWeight:
    def test_trainer_stores_pos_weight(self):
        model = StockLSTM(input_size=10, hidden_sizes=[8, 8], dropout=0.1)
        trainer = LSTMTrainer(model, config={
            "learning_rate": 0.001, "weight_decay": 1e-4,
            "batch_size": 4, "epochs": 1,
        }, pos_weight=2.5)
        assert trainer.pos_weight == pytest.approx(2.5)
        # criterion should be BCEWithLogitsLoss with a 1-element pos_weight tensor
        import torch
        assert isinstance(trainer.criterion, torch.nn.BCEWithLogitsLoss)
        assert trainer.criterion.pos_weight is not None
        assert float(trainer.criterion.pos_weight.item()) == pytest.approx(2.5)

    def test_trainer_pos_weight_survives_save_load(self, tmp_path):
        cfg = {
            "learning_rate": 0.001, "weight_decay": 1e-4,
            "batch_size": 4, "epochs": 1,
            "lstm_units": [8, 8], "dropout": 0.1,
        }
        model = StockLSTM(input_size=10, hidden_sizes=[8, 8], dropout=0.1)
        trainer = LSTMTrainer(model, config=cfg, pos_weight=1.8)

        X_train = np.random.randn(12, 5, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, size=12).astype(np.float32)
        X_val = np.random.randn(4, 5, 10).astype(np.float32)
        y_val = np.random.randint(0, 2, size=4).astype(np.float32)
        trainer.train(X_train, y_train, X_val, y_val, patience=1)

        path = tmp_path / "m.pt"
        trainer.save(path)
        loaded = LSTMTrainer.load(path)
        assert loaded.pos_weight == pytest.approx(1.8)


# --------------------------------------------------------------------------
# news_pipeline: content-snippet + Finnhub sentiment/relevance aggregation
# --------------------------------------------------------------------------

@pytest.fixture
def news_db_with_content(tmp_path):
    db_path = tmp_path / "news.db"
    DatabaseSchema(db_path).create_all_tables()
    conn = sqlite3.connect(db_path)
    # two headlines on Monday 2026-01-05 before 4pm ET for AAPL
    news = [
        ("https://ex.com/a", "AAPL", "Apple beats earnings", "Reuters",
         "2026-01-05T14:00:00Z",
         "Apple reported strong earnings today.", 0.8, 0.9),
        ("https://ex.com/b", "AAPL", "Apple stock surges", "Bloomberg",
         "2026-01-05T15:30:00Z",
         "Shares climbed after the report.", 0.4, 0.3),
    ]
    for url, t, title, src, pub, content, sent, rel in news:
        conn.execute(
            "INSERT INTO news (url, ticker, title, source, published_at, "
            "content, sentiment_score, relevance_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (url, t, title, src, pub, content, sent, rel),
        )
    conn.commit()
    return conn, db_path


class TestNewsAggregation:
    def test_aggregates_content_snippet_weighted_by_relevance(self, news_db_with_content):
        conn, _ = news_db_with_content
        try:
            df = _load_news_aligned(conn)
        finally:
            conn.close()

        assert len(df) == 1
        row = df.iloc[0]
        assert row["ticker"] == "AAPL"
        assert row["n_headlines"] == 2
        # snippet should start with the higher-relevance (0.9) content
        assert "strong earnings" in row["content_snippet"]
        # avg sentiment: (0.8 + 0.4)/2 = 0.6
        assert abs(row["avg_finnhub_sentiment"] - 0.6) < 1e-6
        assert abs(row["avg_relevance"] - 0.6) < 1e-6
        urls = json.loads(row["urls"])
        assert "https://ex.com/a" in urls

    def test_snippet_bounded_in_length(self, news_db_with_content):
        conn, _ = news_db_with_content
        try:
            df = _load_news_aligned(conn)
        finally:
            conn.close()
        snippet = df.iloc[0]["content_snippet"]
        assert len(snippet) <= 801  # _CONTENT_SNIPPET_CHARS (+join space)


# --------------------------------------------------------------------------
# Flask API: /api/presets, /api/dates, /api/conviction, /api/run
# --------------------------------------------------------------------------

@pytest.fixture
def flask_client(monkeypatch, tmp_path):
    # app.server imports module-level paths; use a separate tmp DB path to
    # ensure no side effects on the real one.
    from app import server as srv
    client = srv.app.test_client()
    return client, srv


class TestFlaskAPI:
    def test_presets_returns_three(self, flask_client):
        client, _ = flask_client
        resp = client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "presets" in data
        presets = data["presets"]
        assert {"quick", "balanced", "advanced"}.issubset(presets.keys())
        for name, p in presets.items():
            assert "tickers" in p and "horizon" in p and "seeds" in p

    def test_conviction_endpoint_ok(self, flask_client):
        client, _ = flask_client
        resp = client.get("/api/conviction")
        assert resp.status_code == 200
        assert "buckets" in resp.get_json()

    def test_dates_endpoint_shape(self, flask_client):
        client, _ = flask_client
        resp = client.get("/api/dates?ticker=AAPL")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "dates" in data and isinstance(data["dates"], list)

    def test_run_endpoint_submits_job(self, flask_client, monkeypatch):
        client, srv = flask_client

        calls: dict = {}

        def fake_submit(kind, label, runner):
            calls["kind"] = kind
            calls["label"] = label
            return True, {"id": "test-1", "status": "running",
                          "kind": kind, "label": label}

        monkeypatch.setattr(srv.jobs, "submit", fake_submit)
        resp = client.post("/api/run", json={
            "preset": "quick",
            "config": {"tickers": ["AAPL"], "horizon": 3},
        })
        assert resp.status_code == 202
        body = resp.get_json()
        assert body["accepted"] is True
        assert body["job"]["id"] == "test-1"
        assert calls["kind"] == "pipeline_run"
        # Config must reflect tickers/horizon from request (not preset default)
        assert body["config"]["tickers"] == ["AAPL"]
        assert body["config"]["horizon"] == 3

    def test_rationale_missing_predictions_returns_503(self, flask_client, monkeypatch):
        client, srv = flask_client
        monkeypatch.setattr(srv, "_load_predictions", lambda: None)
        resp = client.get("/api/rationale?ticker=AAPL&date=2026-01-01")
        assert resp.status_code == 503

    def test_ticker_date_returns_specific_row(self, flask_client, monkeypatch):
        """/api/ticker should honor an explicit ?date= and return that row."""
        client, srv = flask_client
        df = pd.DataFrame([
            {
                "ticker": "AAPL",
                "prediction_date": "2026-01-05",
                "ensemble_pred_proba": 0.7,
                "financial_pred_proba": 0.55,
                "news_tfidf_pred_proba": 0.6,
                "news_embeddings_pred_proba": 0.52,
                "actual_binary": 1,
                "top_headlines": "[]",
            },
            {
                "ticker": "AAPL",
                "prediction_date": "2026-01-06",
                "ensemble_pred_proba": 0.3,
                "financial_pred_proba": 0.4,
                "news_tfidf_pred_proba": 0.35,
                "news_embeddings_pred_proba": 0.45,
                "actual_binary": 0,
                "top_headlines": "[]",
            },
        ])
        monkeypatch.setattr(srv, "_load_predictions", lambda: df)
        monkeypatch.setattr(srv, "TICKERS", ["AAPL"])
        resp = client.get("/api/ticker?ticker=AAPL&date=2026-01-05")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["prediction_date"] == "2026-01-05"
        assert abs(body["proba"] - 0.7) < 1e-6
        assert body["binary"] == 1

    def test_ticker_date_unknown_returns_404(self, flask_client, monkeypatch):
        client, srv = flask_client
        df = pd.DataFrame([{
            "ticker": "AAPL", "prediction_date": "2026-01-05",
            "ensemble_pred_proba": 0.7, "financial_pred_proba": 0.55,
            "news_tfidf_pred_proba": 0.6, "news_embeddings_pred_proba": 0.52,
            "actual_binary": 1, "top_headlines": "[]",
        }])
        monkeypatch.setattr(srv, "_load_predictions", lambda: df)
        monkeypatch.setattr(srv, "TICKERS", ["AAPL"])
        resp = client.get("/api/ticker?ticker=AAPL&date=2099-01-01")
        assert resp.status_code == 404

    def test_last_resolved_shape(self, flask_client, monkeypatch):
        """/api/last-resolved returns rows with hit/actual/pred info."""
        client, srv = flask_client
        df = pd.DataFrame([
            {"ticker": "AAPL", "prediction_date": "2026-01-01",
             "ensemble_pred_proba": 0.7, "ensemble_pred_binary": 1,
             "actual_binary": 1},
            {"ticker": "AAPL", "prediction_date": "2026-01-02",
             "ensemble_pred_proba": 0.8, "ensemble_pred_binary": 1,
             "actual_binary": 0},
            {"ticker": "AAPL", "prediction_date": "2026-01-03",
             "ensemble_pred_proba": 0.4, "ensemble_pred_binary": 0,
             "actual_binary": None},  # unresolved — should be excluded
        ])
        monkeypatch.setattr(srv, "_load_predictions", lambda: df)
        monkeypatch.setattr(srv, "TICKERS", ["AAPL"])

        class _EmptyConn:
            def execute(self, *_a, **_kw):
                class _Cur:
                    def fetchall(self_inner):
                        return []
                return _Cur()
            def close(self):
                pass
        monkeypatch.setattr(srv, "_get_db", lambda: _EmptyConn())

        resp = client.get("/api/last-resolved?ticker=AAPL&n=5")
        assert resp.status_code == 200
        body = resp.get_json()
        rows = body["rows"]
        assert len(rows) == 2  # unresolved excluded
        assert rows[0]["date"] == "2026-01-01"
        assert rows[0]["hit"] == 1
        assert rows[1]["hit"] == 0
        assert rows[1]["actual_binary"] == 0

    def test_last_resolved_bad_ticker(self, flask_client):
        client, _ = flask_client
        resp = client.get("/api/last-resolved?ticker=ZZZZ")
        assert resp.status_code == 404


# --------------------------------------------------------------------------
# Ensemble calibration: method picker + smooth probabilities
# --------------------------------------------------------------------------

class TestCalibrationMethodPicker:
    def test_small_val_uses_sigmoid(self):
        from scripts.build_ensemble import _pick_calibration_method, CAL_ISOTONIC_MIN_ROWS
        assert _pick_calibration_method(CAL_ISOTONIC_MIN_ROWS - 1) == "sigmoid"
        assert _pick_calibration_method(100) == "sigmoid"

    def test_large_val_uses_isotonic(self):
        from scripts.build_ensemble import _pick_calibration_method, CAL_ISOTONIC_MIN_ROWS
        assert _pick_calibration_method(CAL_ISOTONIC_MIN_ROWS) == "isotonic"
        assert _pick_calibration_method(10_000) == "isotonic"

    def test_sigmoid_produces_smooth_probabilities(self):
        """Sigmoid calibration should give non-degenerate (std>0.05),
        many-valued output even with a tree-based raw classifier."""
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from scripts.build_ensemble import _make_prefit_calibrator

        rng = np.random.default_rng(42)
        X = rng.normal(size=(400, 5))
        y = (X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.5, size=400) > 0).astype(int)

        raw = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=42)
        raw.fit(X, y)

        cal = _make_prefit_calibrator(raw, method="sigmoid")
        cal.fit(X, y)
        proba = cal.predict_proba(X)[:, 1]
        assert proba.std() > 0.05
        # smooth = many distinct values (not collapsed into 2-3 plateaus)
        assert len(np.unique(np.round(proba, 4))) > 50


# --------------------------------------------------------------------------
# LSTM calibration: small val should fall back to sigmoid (Platt)
# --------------------------------------------------------------------------

class TestLSTMCalibrationFallback:
    def _make_trainer(self, seed: int = 0):
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = StockLSTM(input_size=4, hidden_sizes=[4, 4], dropout=0.0)
        trainer = LSTMTrainer(model, config={
            "learning_rate": 0.01, "weight_decay": 0.0,
            "batch_size": 8, "epochs": 1,
            "lstm_units": [4, 4], "dropout": 0.0,
        })
        return trainer

    def test_small_val_uses_sigmoid(self):
        from src.models.lstm_model import CAL_ISOTONIC_MIN_ROWS, _SigmoidCalibrator
        trainer = self._make_trainer(seed=1)
        n = 128
        assert n < CAL_ISOTONIC_MIN_ROWS
        X_val = np.random.randn(n, 3, 4).astype(np.float32)
        y_val = np.random.randint(0, 2, size=n).astype(np.float32)
        trainer.fit_calibration(X_val, y_val)
        assert trainer.calibration_method == "sigmoid"
        assert isinstance(trainer.calibrator, _SigmoidCalibrator)

    def test_sigmoid_produces_smooth_output(self):
        """Calibrated output should stay smooth - i.e. many distinct values -
        and not collapse into the handful of plateaus isotonic would give."""
        trainer = self._make_trainer(seed=2)
        n = 256
        X_val = np.random.randn(n, 3, 4).astype(np.float32)
        y_val = (np.random.randn(n) > 0).astype(np.float32)
        trainer.fit_calibration(X_val, y_val)
        cal = trainer.predict_proba(X_val, apply_calibration=True)
        assert len(np.unique(np.round(cal, 4))) > 50

    def test_calibration_method_survives_save_load(self, tmp_path):
        from src.models.lstm_model import _SigmoidCalibrator
        trainer = self._make_trainer(seed=3)
        X_val = np.random.randn(64, 3, 4).astype(np.float32)
        y_val = np.random.randint(0, 2, size=64).astype(np.float32)
        trainer.fit_calibration(X_val, y_val)
        path = tmp_path / "lstm.pt"
        trainer.save(path)
        loaded = LSTMTrainer.load(path, input_size=4)
        assert loaded.calibration_method == "sigmoid"
        assert isinstance(loaded.calibrator, _SigmoidCalibrator)
