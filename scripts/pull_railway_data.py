#!/usr/bin/env python3
"""Download /data artifacts from the running Railway container to local ``data/``.

Uses ``railway ssh`` to stream a tarball from the container volume. Required
before running ``daily_update.py`` on GitHub Actions (or locally) when the
canonical DB + models live on Railway.

Usage
-----
  python scripts/pull_railway_data.py
  python scripts/pull_railway_data.py --db-only
  python scripts/pull_railway_data.py --service web
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402

RAILWAY_DATA_ROOT = os.getenv("RAILWAY_DATA_ROOT", "/data")


def _railway_cli() -> str:
    cli = shutil.which("railway")
    if not cli:
        raise RuntimeError(
            "Railway CLI not found. Install: npm install -g @railway/cli"
        )
    return cli


def _ssh_prefix(cli: str, service: str | None) -> list[str]:
    cmd = [cli, "ssh"]
    for key, env in (
        ("-p", "RAILWAY_PROJECT_ID"),
        ("-s", "RAILWAY_SERVICE"),
        ("-e", "RAILWAY_ENVIRONMENT"),
    ):
        val = os.getenv(env if key != "-s" else "RAILWAY_SERVICE")
        if key == "-s" and service:
            val = service
        if val:
            cmd.extend([key, val])
    key_path = os.getenv("RAILWAY_SSH_KEY_PATH")
    if key_path:
        cmd.extend(["-i", key_path])
    return cmd


def pull_from_railway(
    dest: Path,
    *,
    service: str | None = None,
    db_only: bool = False,
) -> None:
    cli = _railway_cli()
    root = RAILWAY_DATA_ROOT.rstrip("/")
    ssh = _ssh_prefix(cli, service)

    if db_only:
        remote_paths = "database.db"
    else:
        remote_paths = "database.db processed models"

    print(f"[pull] Streaming {root}/{{{remote_paths}}} from Railway …")
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        remote_cmd = f"cd {shlex.quote(root)} && tar czf - {remote_paths} 2>/dev/null"
        with open(tmp_path, "wb") as out:
            result = subprocess.run(
                ssh + ["--", "sh", "-c", remote_cmd],
                stdout=out,
                stderr=subprocess.PIPE,
            )
        if result.returncode != 0:
            err = (result.stderr or b"").decode(errors="replace").strip()
            raise RuntimeError(f"railway ssh pull failed: {err or result.returncode}")

        if tmp_path.stat().st_size < 100:
            raise RuntimeError(
                "Pull returned an empty archive — is the service Online and "
                f"the volume mounted at {root}/?"
            )

        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=dest)
        print(f"[pull] Extracted to {dest}")
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull /data from Railway to local data/")
    parser.add_argument(
        "--dest",
        default=str(DATA_DIR),
        help="Local data root (default: repo data/)",
    )
    parser.add_argument(
        "--service",
        default=os.getenv("RAILWAY_SERVICE"),
        help="Railway service name (default: RAILWAY_SERVICE env)",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Only download database.db",
    )
    args = parser.parse_args()
    pull_from_railway(Path(args.dest), service=args.service, db_only=args.db_only)
    print("Done.")


if __name__ == "__main__":
    main()
