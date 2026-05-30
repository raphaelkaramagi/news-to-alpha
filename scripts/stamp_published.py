#!/usr/bin/env python3
"""Write last_published.json after an in-container daily update (Railway /data)."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR  # noqa: E402


def main() -> None:
    source = os.getenv("PUBLISH_SOURCE", "railway_daily_update")
    stamp = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "bundle_version": "1",
    }
    out = PROCESSED_DATA_DIR / "last_published.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stamp, indent=2))
    print(f"Stamped {out} ({source})")


if __name__ == "__main__":
    main()
