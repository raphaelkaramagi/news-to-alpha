#!/usr/bin/env python3
"""Build and upload the deploy bundle to Railway (or a local directory).

Selects only the artifacts needed to serve the API (no training data, no
raw downloads, no tests). Trims the SQLite DB to the last 180 trading days
per ticker to keep the bundle small.

Outputs to deploy_bundle/ (local) or uploads to Railway via CLI.

Usage
-----
  python scripts/publish_deploy_bundle.py --dry-run
  python scripts/publish_deploy_bundle.py --target local --output deploy_bundle/
  python scripts/publish_deploy_bundle.py --target railway

Environment
-----------
  RAILWAY_TOKEN     - required for --target railway (set via Railway CLI or env)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    DATABASE_PATH,
)

# Files to include (relative to DATA_DIR root)
BUNDLE_FILES = [
    # Predictions & evaluation CSVs
    ("processed/final_ensemble_predictions.csv", True),   # (rel_path, required)
    ("processed/evaluation_overall.csv",         False),
    ("processed/evaluation_by_ticker.csv",       False),
    ("processed/evaluation_by_confidence.csv",   False),
    ("processed/pipeline_config.json",           False),
    # Model files needed for rationale + cloud infer
    ("models/ensemble_meta.joblib",              False),
    ("models/lstm_model.pt",                     False),
    ("models/nlp_baseline.joblib",               False),
    ("models/news_tfidf.joblib",                 False),
    ("models/tfidf_lr.joblib",                   False),
    ("models/news_embeddings.joblib",            False),
]

TRIM_DAYS = 180  # keep last N calendar days in the DB per ticker


def _human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def build_manifest(source_data: Path, dry_run: bool) -> list[dict]:
    rows = []
    for rel, required in BUNDLE_FILES:
        src = source_data / rel
        rows.append({
            "path": rel,
            "exists": src.exists(),
            "required": required,
            "size": _human_size(src),
        })
    return rows


def copy_artifacts(source_data: Path, dest: Path, dry_run: bool) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel, required in BUNDLE_FILES:
        src = source_data / rel
        if not src.exists():
            if required:
                raise FileNotFoundError(
                    f"Required artifact missing: {src}\n"
                    "Run `python scripts/run_pipeline.py --preset balanced` first."
                )
            print(f"  [skip]   {rel}")
            continue
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [copy]   {rel}  ({_human_size(src)})")
        if not dry_run:
            shutil.copy2(src, dst)
        copied += 1
    return copied


def trim_database(src_db: Path, dest_db: Path, trim_days: int, dry_run: bool) -> None:
    """Copy src_db to dest_db then DELETE rows older than trim_days per ticker."""
    print(f"  [db]     Trimming DB to last {trim_days} days → {dest_db.name}")
    if dry_run:
        print(f"           DRY-RUN: would copy {_human_size(src_db)} and trim")
        return

    dest_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_db, dest_db)

    conn = sqlite3.connect(str(dest_db))
    try:
        for table, date_col in [("prices", "date"), ("news", "date(published_at)"),
                                  ("labels", "date")]:
            try:
                conn.execute(
                    f"""
                    DELETE FROM {table}
                    WHERE {date_col} < date('now', '-{trim_days} days')
                    """
                )
            except sqlite3.OperationalError:
                pass  # table might not exist or column differs
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()
    print(f"           DB size after trim: {_human_size(dest_db)}")


def write_manifest_stamp(dest: Path, dry_run: bool) -> None:
    stamp = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "bundle_version": "1",
    }
    stamp_path = dest / "processed" / "last_published.json"
    print(f"  [stamp]  {stamp_path.relative_to(dest)}")
    if not dry_run:
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(stamp, indent=2))


def upload_to_railway(bundle_dir: Path) -> None:
    """Upload via Railway CLI using `railway run`."""
    cli = shutil.which("railway")
    if not cli:
        raise RuntimeError(
            "Railway CLI not found. Install: npm install -g @railway/cli\n"
            "Then: railway login"
        )

    print("\n[railway] Uploading bundle ...")
    # Copy each file via `railway run -- cp <local> /data/...`
    for rel, _ in BUNDLE_FILES:
        local = bundle_dir / rel
        if not local.exists():
            continue
        remote = f"/data/{rel}"
        # Ensure remote parent exists
        parent = str(Path(remote).parent)
        subprocess.run(
            [cli, "run", "--", "mkdir", "-p", parent],
            check=False, capture_output=True,
        )
        result = subprocess.run(
            [cli, "run", "--", "cp", "-f", str(local), remote],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [warn] Failed to upload {rel}: {result.stderr.strip()}")
        else:
            print(f"  [ok]   {rel}")

    # Upload stamp
    stamp_local = bundle_dir / "processed" / "last_published.json"
    if stamp_local.exists():
        subprocess.run(
            [cli, "run", "--", "cp", "-f",
             str(stamp_local), "/data/processed/last_published.json"],
            check=False,
        )

    # Upload DB
    db_local = bundle_dir / "database.db"
    if db_local.exists():
        subprocess.run(
            [cli, "run", "--", "cp", "-f", str(db_local), "/data/database.db"],
            check=False,
        )
        print("  [ok]   database.db")

    print("[railway] Upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and publish deploy bundle")
    parser.add_argument("--target", choices=["local", "railway"], default="local",
                        help="Where to put the bundle (default: local)")
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "deploy_bundle"),
                        help="Local output directory (default: deploy_bundle/)")
    parser.add_argument("--trim-days", type=int, default=TRIM_DAYS,
                        help=f"Keep last N days in SQLite (default: {TRIM_DAYS})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print manifest and exit without writing files.")
    args = parser.parse_args()

    print("=" * 60)
    print("DEPLOY BUNDLE")
    print("=" * 60)
    print(f"  source data   : {DATA_DIR}")
    print(f"  target        : {args.target}")
    print(f"  output        : {args.output}")
    print(f"  trim_days     : {args.trim_days}")
    print(f"  dry_run       : {args.dry_run}")
    print()

    manifest = build_manifest(DATA_DIR, args.dry_run)
    print("Manifest:")
    total_size = 0
    for row in manifest:
        status = "OK" if row["exists"] else ("MISSING (req)" if row["required"] else "missing")
        print(f"  {status:14s}  {row['size']:10s}  {row['path']}")
    print()

    if args.dry_run:
        db_size = _human_size(DATABASE_PATH) if DATABASE_PATH.exists() else "missing"
        print(f"  database.db (trimmed to {args.trim_days}d): {db_size} → estimate ~10–30 MB")
        print("\nDRY-RUN complete. No files written.")
        return

    dest = Path(args.output)
    n = copy_artifacts(DATA_DIR, dest, args.dry_run)
    if DATABASE_PATH.exists():
        trim_database(DATABASE_PATH, dest / "database.db", args.trim_days, args.dry_run)
    write_manifest_stamp(dest, args.dry_run)
    print(f"\nBundle ready: {dest}  ({n} artifacts + db)")

    if args.target == "railway":
        upload_to_railway(dest)

    print("\nDone.")


if __name__ == "__main__":
    main()
