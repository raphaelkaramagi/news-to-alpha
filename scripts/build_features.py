#!/usr/bin/env python3
"""
Build the full feature pipeline: labels -> indicators -> LSTM sequences.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --tickers AAPL TSLA
    python scripts/build_features.py --seq-len 30
"""

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS, PROCESSED_DATA_DIR  # noqa: E402
from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_processing.label_generator import LabelGenerator  # noqa: E402
from src.features.technical_indicators import TechnicalIndicators  # noqa: E402
from src.features.sequence_generator import SequenceGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full feature pipeline")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: all)")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length (default: 60)")
    args = parser.parse_args()

    tickers = args.tickers or TICKERS

    DatabaseSchema().create_all_tables()

    print("=" * 60)
    print("FEATURE PIPELINE")
    print("=" * 60)

    # Step 1: generate labels (safe to re-run, skips duplicates)
    print("\n--- Step 1: Labels ---")
    label_summary = LabelGenerator().generate(tickers)
    print(f"  {label_summary['total_labels']} new labels, "
          f"{label_summary['total_skipped']} already existed")

    # Step 2: compute indicators + build sequences per ticker
    print("\n--- Step 2: Indicators + Sequences ---")
    gen = SequenceGenerator(sequence_length=args.seq_len)

    all_X, all_y, all_dates = [], [], []

    for ticker in tickers:
        X, y, dates = gen.generate(ticker)
        if len(X) == 0:
            print(f"  {ticker:5s}  skipped (not enough data for {gen.seq_len}-day sequences)")
            continue

        print(f"  {ticker:5s}  {len(X)} sequences, {y.mean():.0%} up")
        all_X.append(X)
        all_y.append(y)
        all_dates.extend([(ticker, d) for d in dates])

    if not all_X:
        print("\nNo sequences generated. Need more price data (try --days 90+ when collecting).")
        return

    # Combine all tickers into one dataset
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    # Save to data/processed/
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / "X_sequences.npy", X_combined)
    np.save(PROCESSED_DATA_DIR / "y_labels.npy", y_combined)

    print(f"\n--- Summary ---")
    print(f"  Total sequences: {len(X_combined)}")
    print(f"  Shape: {X_combined.shape}")
    print(f"  Up/Down split: {y_combined.mean():.0%} up / {1 - y_combined.mean():.0%} down")
    print(f"\n  Saved to: data/processed/X_sequences.npy")
    print(f"            data/processed/y_labels.npy")
    print("=" * 60)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    main()
