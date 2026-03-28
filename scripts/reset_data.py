#!/usr/bin/env python3
"""
Reset generated data so you can re-run the pipeline.

By default, preserves the database (which contains collected news articles
that take time to accumulate). Use --full to delete everything.

Usage:
    python scripts/reset_data.py              # reset features + models (keep DB)
    python scripts/reset_data.py --full       # delete everything including database
    python scripts/reset_data.py --keep-raw   # also keep raw download files
"""

import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, DATABASE_PATH, PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR  # noqa: E402


def reset(keep_db: bool = True, keep_raw: bool = False) -> None:
    """Delete generated data files.

    Args:
        keep_db: If True (default), preserve the database so accumulated
                 news articles aren't lost.  The demo will re-collect
                 prices and re-generate labels into the existing DB.
        keep_raw: If True, also keep raw download files.
    """
    removed = []

    if not keep_db and DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        removed.append(f"  {DATABASE_PATH.relative_to(DATA_DIR.parent)}")

    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
        removed.append(f"  {PROCESSED_DATA_DIR.relative_to(DATA_DIR.parent)}/")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
        removed.append(f"  {MODELS_DIR.relative_to(DATA_DIR.parent)}/")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not keep_raw and RAW_DATA_DIR.exists():
        shutil.rmtree(RAW_DATA_DIR)
        removed.append(f"  {RAW_DATA_DIR.relative_to(DATA_DIR.parent)}/")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DATA_DIR / "prices").mkdir(exist_ok=True)
        (RAW_DATA_DIR / "news").mkdir(exist_ok=True)

    if removed:
        print("Removed:")
        for r in removed:
            print(r)
        if keep_db:
            print(f"\n  Preserved: {DATABASE_PATH.relative_to(DATA_DIR.parent)} "
                  "(news articles kept)")
    else:
        print("Nothing to remove — already clean.")

    print("\nReady for a fresh run:")
    print("  python scripts/demo.py            # full pipeline demo")
    print("  python scripts/demo.py --reset     # reset + demo in one command")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset generated data")
    parser.add_argument("--full", action="store_true",
                        help="Delete everything including database (loses collected news)")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Also keep raw download files")
    args = parser.parse_args()

    label = "FULL RESET" if args.full else "RESETTING (keeping database)"
    print("=" * 50)
    print(label)
    print("=" * 50)

    reset(keep_db=not args.full, keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
