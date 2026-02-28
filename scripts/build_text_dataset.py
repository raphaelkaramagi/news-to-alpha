#!/usr/bin/env python3
"""
Build the labeled text dataset for the NLP pipeline.

Usage
-----
    python scripts/build_text_dataset.py
    python scripts/build_text_dataset.py --news-db data/news.db
    python scripts/build_text_dataset.py --no-require-labels
    python scripts/build_text_dataset.py --output data/processed/my_dataset.csv
    python scripts/build_text_dataset.py --preview 10
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR  # noqa: E402
from src.data_processing.news_dataset_builder import NewsDatasetBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled text dataset from news headlines + labels"
    )
    parser.add_argument(
        "--db", default=str(DATABASE_PATH),
        help=f"Path to main SQLite database (default: {DATABASE_PATH})",
    )
    parser.add_argument(
        "--news-db", default=None,
        help="Optional path to supplementary news.db (articles table)",
    )
    parser.add_argument(
        "--output", default=None,
        help=f"Output CSV path (default: {PROCESSED_DATA_DIR / 'text_dataset.csv'})",
    )
    parser.add_argument(
        "--no-require-labels", action="store_true",
        help="Include all rows even when labels are absent (left join)",
    )
    parser.add_argument(
        "--preview", type=int, default=5, metavar="N",
        help="Number of sample rows to print (default: 5, 0 = off)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    print("=" * 70)
    print("BUILD TEXT DATASET")
    print("=" * 70)
    print(f"Main DB  : {args.db}")
    if args.news_db:
        print(f"News DB  : {args.news_db}")
    print(f"Labels   : {'required (inner join)' if not args.no_require_labels else 'optional (left join)'}")
    print()

    builder = NewsDatasetBuilder(db_path=args.db, news_db_path=args.news_db)
    df = builder.build(require_labels=not args.no_require_labels)

    if df.empty:
        print("No rows produced — check that news and labels tables are populated.")
        print("\nTip: run  python scripts/collect_news.py  to populate news,")
        print("     and   python scripts/generate_labels.py  for labels.")
        sys.exit(0)

    out_path = builder.save(df, args.output)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows      : {len(df):,}")
    print(f"Tickers         : {df['ticker'].nunique()} ({', '.join(sorted(df['ticker'].unique()))})")
    print(f"Date range      : {df['prediction_date'].min()} -> {df['prediction_date'].max()}")
    print(f"Total articles  : {df['num_articles'].sum():,}  (avg {df['num_articles'].mean():.1f} / day per ticker)")

    labeled = df["label_binary"].notna().sum()
    print(f"Labeled rows    : {labeled:,} / {len(df):,}  ({100 * labeled / len(df):.1f}%)")

    if labeled > 0:
        up = (df["label_binary"] == 1).sum()
        down = (df["label_binary"] == 0).sum()
        print(f"Label balance   : {up} up  /  {down} down  ({100 * up / labeled:.1f}% positive)")

    print(f"\nSaved to: {out_path}")

    if args.preview > 0 and len(df) > 0:
        import pandas as pd
        print(f"\n--- First {min(args.preview, len(df))} rows ---")
        preview_cols = ["ticker", "prediction_date", "num_articles", "label_binary", "headlines_text"]
        preview_cols = [c for c in preview_cols if c in df.columns]
        pd_preview = df[preview_cols].head(args.preview).copy()
        if "headlines_text" in pd_preview.columns:
            pd_preview["headlines_text"] = pd_preview["headlines_text"].str[:80] + "..."
        print(pd_preview.to_string(index=False))

    print("=" * 70)


if __name__ == "__main__":
    main()
