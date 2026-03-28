#!/usr/bin/env python3
"""
Reset all generated data so you can test the pipeline from scratch.

Usage:
    python scripts/reset_data.py              # delete everything
    python scripts/reset_data.py --keep-raw   # keep downloaded prices/news, reset features + models
"""

import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, DATABASE_PATH, PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR  # noqa: E402


def reset(keep_raw: bool = False) -> None:
    """Delete generated data files. Optionally keep raw downloads."""

    removed = []

    # Database
    if DATABASE_PATH.exists():
        DATABASE_PATH.unlink()
        removed.append(f"  {DATABASE_PATH.relative_to(DATA_DIR.parent)}")

    # Processed features (sequences, splits, dates)
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
        removed.append(f"  {PROCESSED_DATA_DIR.relative_to(DATA_DIR.parent)}/")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Trained models
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
        removed.append(f"  {MODELS_DIR.relative_to(DATA_DIR.parent)}/")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Raw data (optional)
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
    else:
        print("Nothing to remove — already clean.")

    print("\nReady for a fresh run:")
    print("  python scripts/demo.py            # full pipeline demo")
    print("  python scripts/demo.py --reset     # reset + demo in one command")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset all generated data")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep downloaded prices/news, only reset features + models")
    args = parser.parse_args()

    print("=" * 50)
    print("RESETTING DATA" + (" (keeping raw)" if args.keep_raw else ""))
    print("=" * 50)

    reset(keep_raw=args.keep_raw)


if __name__ == "__main__":
    main()
