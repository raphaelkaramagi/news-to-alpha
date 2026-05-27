"""Tests for Phase 0 additions:

- pipeline_config.py: save/load round-trip, load_or_default fallback
- trading_calendar.py: last_trading_session, sessions_behind
- publish_deploy_bundle.py: manifest building, dry-run
- Flask /api/data-status: new fields (last_trading_session, is_current, trading_sessions_behind)
- Flask INFERENCE_ONLY guard
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# pipeline_config
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_save_and_load(self, tmp_path):
        from src.utils.pipeline_config import save, load
        cfg = {"horizon": 3, "seeds": [42, 1337], "tickers": ["AAPL", "NVDA"]}
        p = tmp_path / "pipeline_config.json"
        save(cfg, path=p)
        loaded = load(path=p)
        assert loaded is not None
        assert loaded["horizon"] == 3
        assert loaded["seeds"] == [42, 1337]
        assert "saved_at" in loaded
        assert loaded["run_type"] == "full_train"

    def test_load_returns_none_when_missing(self, tmp_path):
        from src.utils.pipeline_config import load
        result = load(path=tmp_path / "nonexistent.json")
        assert result is None

    def test_load_or_default_returns_balanced(self, tmp_path):
        from src.utils.pipeline_config import load_or_default
        result = load_or_default(path=tmp_path / "nonexistent.json")
        assert result["horizon"] == 3
        assert "seeds" in result
        assert len(result["tickers"]) > 0

    def test_save_preserves_run_type(self, tmp_path):
        from src.utils.pipeline_config import save, load
        p = tmp_path / "cfg.json"
        save({"horizon": 1}, path=p, run_type="daily_update")
        loaded = load(path=p)
        assert loaded["run_type"] == "daily_update"


# ---------------------------------------------------------------------------
# trading_calendar
# ---------------------------------------------------------------------------

class TestTradingCalendar:
    def test_last_trading_session_is_not_today(self):
        from src.utils.trading_calendar import last_trading_session
        today = date.today()
        last = last_trading_session(as_of=today)
        assert last < today

    def test_last_trading_session_not_weekend(self):
        from src.utils.trading_calendar import last_trading_session
        last = last_trading_session()
        assert last.weekday() < 5  # Mon–Fri

    def test_sessions_behind_zero_when_current(self):
        from src.utils.trading_calendar import sessions_behind, last_trading_session
        # If we feed the last trading session minus horizon=1 as latest_pred
        last = last_trading_session()
        one_before = last - timedelta(days=7)
        # Go back 1 weekday from last
        d = last - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        # With horizon=1, expected = last - 1 session; if latest_pred == that, behind == 0
        behind = sessions_behind(d.isoformat(), horizon=1)
        assert behind == 0

    def test_sessions_behind_positive_when_stale(self):
        from src.utils.trading_calendar import sessions_behind
        # A date 30 days ago should be many sessions behind
        old_date = (date.today() - timedelta(days=30)).isoformat()
        behind = sessions_behind(old_date, horizon=1)
        assert behind > 0

    def test_sessions_behind_none_returns_minus_one(self):
        from src.utils.trading_calendar import sessions_behind
        assert sessions_behind(None) == -1


# ---------------------------------------------------------------------------
# publish_deploy_bundle: manifest
# ---------------------------------------------------------------------------

class TestPublishBundle:
    def test_build_manifest_returns_rows(self, tmp_path):
        from scripts.publish_deploy_bundle import build_manifest
        rows = build_manifest(source_data=tmp_path, dry_run=True)
        assert isinstance(rows, list)
        assert len(rows) > 0
        # final_ensemble_predictions.csv is required
        req = [r for r in rows if r["required"]]
        assert len(req) == 1
        assert "final_ensemble_predictions.csv" in req[0]["path"]

    def test_dry_run_does_not_write(self, tmp_path):
        from scripts.publish_deploy_bundle import build_manifest
        rows = build_manifest(source_data=tmp_path, dry_run=True)
        # Nothing written to tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_copy_artifacts_skips_missing_optional(self, tmp_path):
        from scripts.publish_deploy_bundle import copy_artifacts
        # Required file missing → raises; but optional missing → just skip
        # Create only the required file
        src = tmp_path / "src"
        src.mkdir()
        (src / "processed").mkdir()
        (src / "processed" / "final_ensemble_predictions.csv").write_text("ticker\n")
        dst = tmp_path / "dst"
        copied = copy_artifacts(src, dst, dry_run=False)
        assert copied >= 1  # at minimum the required file
        assert (dst / "processed" / "final_ensemble_predictions.csv").exists()


# ---------------------------------------------------------------------------
# Flask /api/data-status new fields
# ---------------------------------------------------------------------------

@pytest.fixture
def flask_client_with_predictions(monkeypatch, tmp_path):
    """Flask test client with a minimal predictions CSV."""
    from app import server as srv
    # Point paths to tmp_path
    monkeypatch.setattr(srv, "PREDICTIONS_CSV", tmp_path / "preds.csv")
    monkeypatch.setattr(srv, "EVAL_OVERALL_CSV", tmp_path / "eo.csv")
    monkeypatch.setattr(srv, "EVAL_BY_TICKER_CSV", tmp_path / "ebt.csv")
    monkeypatch.setattr(srv, "DATABASE_PATH", tmp_path / "test.db")

    # Write a minimal CSV
    df = pd.DataFrame([{
        "ticker": "AAPL",
        "prediction_date": "2026-05-20",
        "ensemble_pred_proba": 0.6,
        "ensemble_pred_binary": 1,
        "actual_binary": 1,
    }])
    df.to_csv(tmp_path / "preds.csv", index=False)

    # Empty DB
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.execute("CREATE TABLE IF NOT EXISTS prices (ticker TEXT, date TEXT, close REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS news (published_at TEXT)")
    conn.commit()
    conn.close()

    client = srv.app.test_client()
    return client, srv


class TestDataStatus:
    def test_new_fields_present(self, flask_client_with_predictions):
        client, _ = flask_client_with_predictions
        resp = client.get("/api/data-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "last_trading_session" in data
        assert "trading_sessions_behind" in data
        assert "is_current" in data
        assert "deploy_mode" in data

    def test_deploy_mode_full_by_default(self, flask_client_with_predictions, monkeypatch):
        client, srv = flask_client_with_predictions
        monkeypatch.setattr(srv, "_INFERENCE_ONLY", False)
        resp = client.get("/api/data-status")
        assert resp.get_json()["deploy_mode"] == "full"

    def test_deploy_mode_inference_when_flag_set(self, flask_client_with_predictions, monkeypatch):
        client, srv = flask_client_with_predictions
        monkeypatch.setattr(srv, "_INFERENCE_ONLY", True)
        resp = client.get("/api/data-status")
        assert resp.get_json()["deploy_mode"] == "inference_only"


# ---------------------------------------------------------------------------
# INFERENCE_ONLY guard
# ---------------------------------------------------------------------------

class TestInferenceOnlyGuard:
    def test_api_run_blocked(self, flask_client_with_predictions, monkeypatch):
        client, srv = flask_client_with_predictions
        monkeypatch.setattr(srv, "_INFERENCE_ONLY", True)
        resp = client.post("/api/run", json={"preset": "quick"})
        assert resp.status_code == 403

    def test_api_train_blocked(self, flask_client_with_predictions, monkeypatch):
        client, srv = flask_client_with_predictions
        monkeypatch.setattr(srv, "_INFERENCE_ONLY", True)
        resp = client.post("/api/train", json={"model": "lstm"})
        assert resp.status_code == 403

    def test_api_reset_blocked(self, flask_client_with_predictions, monkeypatch):
        client, srv = flask_client_with_predictions
        monkeypatch.setattr(srv, "_INFERENCE_ONLY", True)
        resp = client.post("/api/reset", json={"model": "lstm"})
        assert resp.status_code == 403

    def test_api_refresh_allowed_in_inference_mode(self, flask_client_with_predictions, monkeypatch):
        client, srv = flask_client_with_predictions
        monkeypatch.setattr(srv, "_INFERENCE_ONLY", True)
        # Mock job submission to avoid actually running
        monkeypatch.setattr(srv.jobs, "submit", lambda **kw: (True, {"id": "x", "status": "running"}))
        resp = client.post("/api/data/refresh", json={"days": 30})
        # 202 accepted (or 409 conflict); either is fine — just not 403
        assert resp.status_code in (202, 409)
