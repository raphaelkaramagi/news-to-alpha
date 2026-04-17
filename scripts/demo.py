#!/usr/bin/env python3
"""Full end-to-end demo: collect data -> build features -> train models -> ensemble.

A single command that verifies the entire pipeline works. Under the hood it
delegates all training to the canonical scripts (`train_lstm.py`,
`train_nlp.py`, `train_news_embeddings.py`, `build_ensemble.py`,
`evaluate_predictions.py`) via `subprocess.run` - so there's a single source
of truth for training logic.

Usage
-----
    python scripts/demo.py                         # all 15 tickers, 250 days
    python scripts/demo.py --reset                 # wipe first, then run
    python scripts/demo.py --tickers AAPL NVDA     # pick your own
    python scripts/demo.py --days 365              # more history
    python scripts/demo.py --quick                 # 2 tickers, fast sanity check
    python scripts/demo.py --skip-training         # data pipeline only
    python scripts/demo.py --skip-news             # no Finnhub API calls
    python scripts/demo.py --skip-embeddings       # skip MiniLM (no HF download)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_collection.price_collector import PriceCollector  # noqa: E402
from src.data_collection.news_collector import NewsCollector  # noqa: E402
from src.data_processing.price_validation import PriceDataValidator  # noqa: E402
from src.data_processing.label_generator import LabelGenerator  # noqa: E402
from src.data_processing.dataset_split import DatasetSplitter  # noqa: E402
from src.features.sequence_generator import SequenceGenerator  # noqa: E402
from src.config import (  # noqa: E402
    TICKERS, FINNHUB_API_KEY, DATABASE_PATH,
    PROCESSED_DATA_DIR, LSTM_CONFIG, MARKET_INDEX_TICKER,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DEFAULT_DAYS = 250
DEFAULT_TICKERS = TICKERS


def _run(cmd: list[str], label: str) -> bool:
    """Run a subcommand, stream its output, and return True on exit 0."""
    print(f"   $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"   {label} exited with code {result.returncode}")
        return False
    return True


def step_reset() -> None:
    from scripts.reset_data import reset
    print("\n0. Resetting features + models (keeping database)...")
    reset(keep_db=True)
    print("   Clean slate (news articles preserved)")


def step_database() -> str:
    db_path = str(DATABASE_PATH)
    DatabaseSchema(db_path).create_all_tables()
    print("   Database ready (5 tables)")
    return db_path


def step_collect_prices(db_path: str, tickers: list[str],
                        start_date: str, end_date: str) -> dict:
    stats = PriceCollector(db_path).collect(tickers, start_date, end_date)
    print(f"   {stats['rows_added']} price rows "
          f"({stats['duplicates_skipped']} dupes skipped)")
    if stats["tickers_failed"]:
        print(f"   Failed: {', '.join(stats['tickers_failed'])}")
    return stats


def step_collect_news(db_path: str, tickers: list[str],
                      start_date: str, end_date: str) -> dict | None:
    if not FINNHUB_API_KEY:
        print("   Skipped (set NEWS_API_KEY in .env for news)")
        return None
    stats = NewsCollector(FINNHUB_API_KEY, db_path).collect(
        tickers, start_date, end_date
    )
    print(f"   {stats['rows_added']} articles "
          f"({stats['duplicates_skipped']} dupes skipped)")
    return stats


def step_validate(db_path: str, tickers: list[str]) -> None:
    results = PriceDataValidator(db_path).validate(tickers)
    for info in results["coverage"]:
        print(f"   {info['ticker']:5s}  {info['days_collected']} trading days  "
              f"({info['first_date']} -> {info['last_date']})")
    issues = (len(results["missing_data"])
              + len(results["price_anomalies"])
              + len(results["volume_anomalies"]))
    if issues:
        print(f"   {issues} data quality issues found")
    else:
        print("   No data quality issues")


def step_labels(tickers: list[str]) -> None:
    summary = LabelGenerator().generate(tickers)
    total = summary['total_labels'] + summary['total_skipped']
    if summary['total_labels'] > 0:
        print(f"   {summary['total_labels']} new labels "
              f"({summary['total_skipped']} already existed)")
    else:
        print(f"   {total} labels (all already existed)")


def step_split() -> None:
    summary = DatasetSplitter().split()
    for name in ("train", "val", "test"):
        s = summary[name]
        print(f"   {name:5s}  {s['num_days']:3d} days  {s['date_range']}  "
              f"({s['prices']} prices, {s['labels']} labels)")


def step_build_features(tickers: list[str]) -> bool:
    gen = SequenceGenerator()
    all_X, all_y, all_dates = [], [], []

    for ticker in tickers:
        X, y, _returns, dates = gen.generate(ticker)
        if len(X) == 0:
            print(f"   {ticker:5s}  skipped (need >= {gen.seq_len + 1} valid rows "
                  f"after indicator warmup)")
            continue
        print(f"   {ticker:5s}  {len(X)} sequences, {y.mean():.0%} up")
        all_X.append(X)
        all_y.append(y)
        all_dates.extend([(ticker, d) for d in dates])

    if not all_X:
        print("   No sequences - increase --days (need >= 250 for LSTM)")
        return False

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / "X_sequences.npy", X_combined)
    np.save(PROCESSED_DATA_DIR / "y_labels.npy", y_combined)
    with open(PROCESSED_DATA_DIR / "sequence_dates.json", "w") as f:
        json.dump(all_dates, f)

    print(f"\n   {len(X_combined)} total sequences, shape {X_combined.shape}")
    return True


def step_sample_data(db_path: str, tickers: list[str]) -> None:
    conn = sqlite3.connect(db_path)
    ph = ",".join("?" for _ in tickers)

    rows = conn.execute(
        f"SELECT ticker, date, close FROM prices "
        f"WHERE ticker IN ({ph}) ORDER BY date DESC LIMIT 5",
        tickers,
    ).fetchall()
    if rows:
        print("\n   Recent prices:")
        for r in rows:
            print(f"     {r[0]:5s} {r[1]}  ${r[2]:.2f}")

    rows = conn.execute(
        f"SELECT ticker, title FROM news WHERE ticker IN ({ph}) LIMIT 3",
        tickers,
    ).fetchall()
    if rows:
        print("\n   Sample headlines:")
        for r in rows:
            title = r[1][:65] + "..." if len(r[1]) > 65 else r[1]
            print(f"     [{r[0]}] {title}")

    rows = conn.execute(
        f"SELECT ticker, date, label_binary, ROUND(label_return, 2) "
        f"FROM labels WHERE ticker IN ({ph}) ORDER BY date DESC LIMIT 5",
        tickers,
    ).fetchall()
    if rows:
        print("\n   Recent labels:")
        for r in rows:
            direction = "UP" if r[2] == 1 else "DN"
            print(f"     {r[0]:5s} {r[1]}  {direction}  {r[3]:+.2f}%")

    conn.close()


def step_train_lstm(epochs: int) -> bool:
    return _run(
        [PYTHON, "scripts/train_lstm.py", "--epochs", str(epochs)],
        "train_lstm.py",
    )


def step_train_nlp() -> bool:
    return _run([PYTHON, "scripts/train_nlp.py"], "train_nlp.py")


def step_train_embeddings() -> bool:
    return _run(
        [PYTHON, "scripts/train_news_embeddings.py"],
        "train_news_embeddings.py",
    )


def step_build_eval_dataset() -> bool:
    return _run(
        [PYTHON, "scripts/build_eval_dataset.py"],
        "build_eval_dataset.py",
    )


def step_build_ensemble() -> bool:
    return _run([PYTHON, "scripts/build_ensemble.py"], "build_ensemble.py")


def step_evaluate() -> bool:
    return _run(
        [PYTHON, "scripts/evaluate_predictions.py"],
        "evaluate_predictions.py",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full end-to-end pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reset", action="store_true",
                        help="Wipe features + models before running (keeps DB)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help=f"Tickers to use (default: all {len(TICKERS)})")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with just AAPL + TSLA")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Calendar days of price history (default: {DEFAULT_DAYS})")
    parser.add_argument("--epochs", type=int, default=50,
                        help="LSTM training epochs (default: 50)")
    parser.add_argument("--skip-news", action="store_true",
                        help="Skip Finnhub news collection")
    parser.add_argument("--skip-training", action="store_true",
                        help="Data pipeline only, skip model training")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip sentence-embedding model (avoids HF download)")
    parser.add_argument("--skip-ensemble", action="store_true",
                        help="Skip ensemble build + evaluation")
    args = parser.parse_args()

    if args.quick:
        tickers = ["AAPL", "TSLA"]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = DEFAULT_TICKERS

    days = args.days

    print("=" * 70)
    print("NEWS-TO-ALPHA  -  FULL PIPELINE DEMO")
    print("=" * 70)
    print(f"Tickers : {', '.join(tickers)}  ({len(tickers)} total)")
    print(f"History : {days} calendar days")

    if args.reset:
        step_reset()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print("\n1. Setting up database...")
    db_path = step_database()

    price_tickers = list(dict.fromkeys([*tickers, MARKET_INDEX_TICKER]))
    print(f"\n2. Collecting prices ({start_date} -> {end_date}) "
          f"[includes {MARKET_INDEX_TICKER} market regime]...")
    step_collect_prices(db_path, price_tickers, start_date, end_date)

    if not args.skip_news:
        print("\n3. Collecting news...")
        step_collect_news(db_path, tickers, start_date, end_date)
    else:
        print("\n3. Skipping news collection (--skip-news)")

    print("\n4. Validating data...")
    step_validate(db_path, tickers)

    print("\n5. Generating labels...")
    step_labels(tickers)

    print("\n6. Splitting dataset (70 / 15 / 15)...")
    step_split()

    print(f"\n7. Building features "
          f"(indicators -> {LSTM_CONFIG['sequence_length']}-day LSTM sequences)...")
    has_sequences = step_build_features(tickers)

    print("\n8. Sample data...")
    step_sample_data(db_path, tickers)

    if args.skip_training:
        print("\nSkipping training (--skip-training).")
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        return

    if not has_sequences:
        print("\nNot enough data for training. Increase --days.")
        return

    print(f"\n9.  Training LSTM (up to {args.epochs} epochs)...")
    lstm_ok = step_train_lstm(args.epochs)

    print("\n10. Training TF-IDF NLP baseline...")
    nlp_ok = step_train_nlp()

    emb_ok = False
    if not args.skip_embeddings:
        print("\n11. Training sentence-embedding NLP model...")
        emb_ok = step_train_embeddings()
    else:
        print("\n11. Skipping sentence-embedding model (--skip-embeddings)")

    if not args.skip_ensemble and lstm_ok and nlp_ok:
        print("\n12. Joining per-model predictions (build_eval_dataset)...")
        eval_ok = step_build_eval_dataset()
        if eval_ok:
            print("\n13. Building ensemble (learned meta-model)...")
            ensemble_ok = step_build_ensemble()
            if ensemble_ok:
                print("\n14. Evaluating models...")
                step_evaluate()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("  Database : data/database.db")
    print("  Features : data/processed/")
    print("  Models   : data/models/")
    print()
    print("  LSTM trained        :", "yes" if lstm_ok else "no")
    print("  TF-IDF trained      :", "yes" if nlp_ok else "no")
    print("  Embeddings trained  :", "yes" if emb_ok else "skipped/failed")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
