#!/usr/bin/env python3
"""
Split collected data into train / validation / test sets (chronological).

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --train 0.80 --val 0.10
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing.dataset_split import DatasetSplitter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological dataset split")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET SPLIT (chronological)")
    print("=" * 60)
    print(f"Ratios: {args.train:.0%} train / {args.val:.0%} val / {1 - args.train - args.val:.0%} test\n")

    summary = DatasetSplitter().split(train_ratio=args.train, val_ratio=args.val)

    for name, info in summary.items():
        print(f"  {name.upper():5s}  {info['date_range']}")
        print(f"         {info['num_days']} days | {info['prices']} prices | {info['news']} news | {info['labels']} labels")
        print()

    print("Saved to: data/processed/split_info.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
