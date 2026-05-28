"""Helpers to avoid stale artifacts when re-running the training pipeline."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path


def clear_stale_lstm_seed_models(models_dir: Path) -> int:
    """Remove lstm_model_seed{N}.pt files (ignores macOS duplicate names)."""
    removed = 0
    for path in models_dir.glob("lstm_model_seed*.pt"):
        if not re.fullmatch(r"lstm_model_seed\d+\.pt", path.name):
            continue
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def prune_predictions_db(db_path: Path) -> int:
    """Keep one row per (ticker, date) — latest insert wins. Drops old model_version rows."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            DELETE FROM predictions
            WHERE rowid NOT IN (
                SELECT MAX(rowid) FROM predictions GROUP BY ticker, date
            )
            """
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
