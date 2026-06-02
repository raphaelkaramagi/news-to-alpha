#!/usr/bin/env python3
"""Download /data artifacts from a running Railway container to local data/."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATA_DIR, DATABASE_PATH, PROCESSED_DATA_DIR, MODELS_DIR  # noqa: E402

RAILWAY_DATA_ROOT = os.getenv("RAILWAY_DATA_ROOT", "/data")

PULL_PATHS = [
    "database.db",
    "processed/final_ensemble_predictions.csv",
    "processed/price_predictions.csv",
    "processed/news_tfidf_predictions.csv",
    "processed/news_embeddings_predictions.csv",
    "processed/pipeline_config.json",
    "processed/evaluation_overall.csv",
    "processed/evaluation_by_ticker.csv",
    "processed/evaluation_by_confidence.csv",
    "models/lstm_model.pt",
    "models/nlp_baseline.joblib",
    "models/news_embeddings.joblib",
    "models/ensemble_meta.joblib",
]


def _railway_ssh_cmd(cli: str, service: str | None) -> list[str]:
    cmd = [cli, "ssh"]
    if service:
        cmd.extend(["-s", service])
    return cmd


def _download(cli: str, remote: str, local: Path, service: str | None) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    ssh = _railway_ssh_cmd(cli, service)
    result = subprocess.run(
        ssh + ["--", "cat", shlex.quote(remote)],
        capture_output=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or b"").decode(errors="replace").strip()
        print(f"  [skip] {remote}: {err[:120]}")
        return False
    if not result.stdout:
        print(f"  [skip] {remote}: empty")
        return False
    local.write_bytes(result.stdout)
    print(f"  [ok]   {remote} -> {local}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Railway /data into local data/")
    parser.add_argument("--service", default=os.getenv("RAILWAY_SERVICE", "web"))
    parser.add_argument("--db-only", action="store_true", help="Only pull database.db")
    args = parser.parse_args()

    cli = shutil.which("railway")
    if not cli:
        raise RuntimeError("Railway CLI not found. Install: npm install -g @railway/cli")

    root = RAILWAY_DATA_ROOT.rstrip("/")
    paths = ["database.db"] if args.db_only else PULL_PATHS

    print("=" * 60)
    print("PULL RAILWAY DATA")
    print("=" * 60)
    print(f"  remote root : {root}/")
    print(f"  local data  : {DATA_DIR}")
    print(f"  service     : {args.service}")
    print()

    ok = 0
    for rel in paths:
        remote = f"{root}/{rel}"
        local = DATA_DIR / rel
        if _download(cli, remote, local, args.service):
            ok += 1

    print(f"\nPulled {ok}/{len(paths)} files.")
    if DATABASE_PATH.exists():
        import sqlite3
        conn = sqlite3.connect(str(DATABASE_PATH))
        try:
            p = conn.execute("SELECT MAX(date) FROM prices").fetchone()[0]
            l = conn.execute("SELECT MAX(date) FROM labels").fetchone()[0]
            print(f"  prices latest : {p}")
            print(f"  labels latest : {l}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()
