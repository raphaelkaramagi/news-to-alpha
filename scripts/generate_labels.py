#!/usr/bin/env python3
"""
Generate up/down labels from collected price data.

Usage:
    python scripts/generate_labels.py
    python scripts/generate_labels.py --tickers AAPL TSLA
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_processing.label_generator import LabelGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate up/down labels")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: all)")
    args = parser.parse_args()

    tickers = args.tickers or TICKERS

    DatabaseSchema().create_all_tables()

    print("=" * 60)
    print("LABEL GENERATION")
    print("=" * 60)
    print(f"Tickers: {len(tickers)} total\n")

    summary = LabelGenerator().generate(tickers)

    # Print per-ticker breakdown
    for ticker, info in summary["tickers"].items():
        status = f"+{info['labels']} labels"
        if info["skipped_dupes"]:
            status += f", {info['skipped_dupes']} already existed"
        print(f"  {ticker:5s}  {status}")

    print(f"\nTotal: {summary['total_labels']} new labels")
    if summary["total_skipped"]:
        print(f"       {summary['total_skipped']} duplicates skipped")
    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    main()
