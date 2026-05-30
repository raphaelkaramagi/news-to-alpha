"""Persist and load the last successful pipeline configuration.

Written to data/processed/pipeline_config.json by run_pipeline.py and
daily_update.py after each successful run. Read by the cloud infer
fallback and any refresh job so horizon/seeds/finbert are never lost.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PATH = _PROJECT_ROOT / "data" / "processed" / "pipeline_config.json"


def save(
    config: dict,
    path: Optional[Path] = None,
    run_type: str = "full_train",
) -> None:
    """Write config to JSON, merging with existing if present."""
    p = path or _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,
        **config,
    }
    p.write_text(json.dumps(payload, indent=2))


def load(path: Optional[Path] = None) -> Optional[dict]:
    """Return saved config dict, or None if not found."""
    p = path or _DEFAULT_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_or_default(path: Optional[Path] = None) -> dict:
    """Return saved config, or canonical project defaults as a fallback."""
    from src.config import TICKERS

    cfg = load(path)
    if cfg:
        return cfg
    return {
        "tickers": list(TICKERS),
        "lookback_days": 730,
        "horizon": 1,
        "min_move_pct": 0.5,
        "seeds": [42, 1337, 2024],
        "use_finbert": True,
        "encoder_model": "finbert",
        "conditional_ensemble": True,
    }
