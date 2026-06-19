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

# Relative to RAILWAY_DATA_ROOT
PULL_PATHS = ["database.db", "processed", "models"]


def _railway_cli() -> str:
    cli = shutil.which("railway")
    if not cli:
        raise RuntimeError(
            "Railway CLI not found. Install: npm install -g @railway/cli"
        )
    return cli


def _ssh_cmd(cli: str, service: str | None) -> list[str]:
    """Match the flags used by the working daily-update SSH steps in CI."""
    cmd = [cli, "ssh"]
    for flag, env in (
        ("-p", "RAILWAY_PROJECT_ID"),
        ("-s", "RAILWAY_SERVICE"),
        ("-e", "RAILWAY_ENVIRONMENT"),
    ):
        val = os.getenv(env)
        if flag == "-s" and service:
            val = service
        if val:
            cmd.extend([flag, val])
    key_path = os.getenv("RAILWAY_SSH_KEY_PATH")
    if key_path:
        cmd.extend(["-i", key_path])
    return cmd


def _ssh_env() -> dict[str, str]:
    """Railway SSH in CI uses the registered key + -p/-s/-e, not RAILWAY_TOKEN."""
    env = os.environ.copy()
    env.pop("RAILWAY_TOKEN", None)
    return env


def _run_ssh(
    ssh: list[str],
    remote: str,
    *,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess:
    """Run ``railway ssh -- <remote>`` and return the completed process."""
    proc = subprocess.run(
        ssh + ["--", "sh", "-c", remote],
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE,
        env=_ssh_env(),
        check=False,
    )
    return proc


def _ssh_text(ssh: list[str], remote: str) -> str:
    proc = _run_ssh(ssh, remote, capture_stdout=True)
    out = (proc.stdout or b"").decode(errors="replace").strip()
    err = (proc.stderr or b"").decode(errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"railway ssh failed ({proc.returncode}): {err or out or 'no output'}"
        )
    return out


def _existing_paths(ssh: list[str], root: str, candidates: list[str]) -> list[str]:
    """Return which candidate paths exist under ``root`` on the remote host."""
    joined = " ".join(shlex.quote(p) for p in candidates)
    script = (
        f"cd {shlex.quote(root)} 2>/dev/null || {{ echo 'MISSING_ROOT'; exit 2; }}; "
        f"for p in {joined}; do [ -e \"$p\" ] && echo \"$p\"; done"
    )
    out = _ssh_text(ssh, script)
    if out == "MISSING_ROOT":
        raise RuntimeError(f"Volume root {root}/ not found in container")
    found = [line.strip() for line in out.splitlines() if line.strip()]
    return found


def pull_from_railway(
    dest: Path,
    *,
    service: str | None = None,
    db_only: bool = False,
) -> None:
    cli = _railway_cli()
    root = RAILWAY_DATA_ROOT.rstrip("/")
    ssh = _ssh_cmd(cli, service)
    candidates = ["database.db"] if db_only else list(PULL_PATHS)

    print(f"[pull] Probing {root}/ on Railway …")
    listing = _ssh_text(ssh, f"ls -la {shlex.quote(root)} 2>&1 || ls -la /")
    print(listing)

    found = _existing_paths(ssh, root, candidates)
    if not found:
        raise RuntimeError(
            f"Nothing to pull under {root}/ — expected one of: {', '.join(candidates)}"
        )
    print(f"[pull] Found: {', '.join(found)}")

    dest.mkdir(parents=True, exist_ok=True)
    path_args = " ".join(shlex.quote(p) for p in found)
    remote_cmd = f"cd {shlex.quote(root)} && tar czf - {path_args}"

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        print(f"[pull] Streaming tarball …")
        with open(tmp_path, "wb") as out:
            proc = subprocess.run(
                ssh + ["--", "sh", "-c", remote_cmd],
                stdout=out,
                stderr=subprocess.PIPE,
                env=_ssh_env(),
                check=False,
            )
        err = (proc.stderr or b"").decode(errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(
                f"tar over railway ssh failed ({proc.returncode}): {err or 'no stderr'}"
            )

        size = tmp_path.stat().st_size
        if size < 100:
            raise RuntimeError(
                f"Pull returned a tiny archive ({size} B) — SSH may have dropped binary data"
            )
        print(f"[pull] Received {size / (1024 * 1024):.1f} MB")

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
