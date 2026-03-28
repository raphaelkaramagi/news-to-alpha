#!/usr/bin/env python3
"""
Full end-to-end demo: collect data → build features → train models.

The single command to verify the entire pipeline works.  Collects enough
price history for the 60-day LSTM windows (~250 calendar days = ~175
trading days, minus ~34 for indicator warmup = ~140 usable rows).

Usage:
    python scripts/demo.py                          # 2 tickers, 250 days
    python scripts/demo.py --reset                  # wipe first, then run
    python scripts/demo.py --tickers AAPL NVDA TSLA # pick your own
    python scripts/demo.py --days 365               # more history
    python scripts/demo.py --all                    # all 15 tickers
    python scripts/demo.py --all --days 365         # maximum data
    python scripts/demo.py --skip-training          # data pipeline only
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
from datetime import datetime, timedelta

from src.database.schema import DatabaseSchema  # noqa: E402
from src.data_collection.price_collector import PriceCollector  # noqa: E402
from src.data_collection.news_collector import NewsCollector  # noqa: E402
from src.data_processing.price_validation import PriceDataValidator  # noqa: E402
from src.data_processing.label_generator import LabelGenerator  # noqa: E402
from src.data_processing.dataset_split import DatasetSplitter  # noqa: E402
from src.features.sequence_generator import SequenceGenerator  # noqa: E402
from src.config import (  # noqa: E402
    TICKERS, FINNHUB_API_KEY, DATABASE_PATH,
    PROCESSED_DATA_DIR, MODELS_DIR, LSTM_CONFIG,
)

# Use all tickers by default so training has enough samples.
# 2 tickers = 58 training sequences (too few for LSTM to learn).
# 15 tickers = 1,245 training sequences (meaningful baseline).
# 250 calendar days ≈ 175 trading days → ~80 sequences per ticker.
DEFAULT_DAYS = 250
DEFAULT_TICKERS = TICKERS


# ── pipeline steps (each prints its own status) ─────────────────────

def step_reset():
    from scripts.reset_data import reset
    print("\n0. Resetting all data...")
    reset()
    print("   ✓ Clean slate")


def step_database():
    db_path = str(DATABASE_PATH)
    DatabaseSchema(db_path).create_all_tables()
    print("   ✓ Database ready (5 tables)")
    return db_path


def step_collect_prices(db_path, tickers, start_date, end_date):
    stats = PriceCollector(db_path).collect(tickers, start_date, end_date)
    print(f"   ✓ {stats['rows_added']} price rows "
          f"({stats['duplicates_skipped']} dupes skipped)")
    if stats["tickers_failed"]:
        print(f"   ⚠ Failed: {', '.join(stats['tickers_failed'])}")
    return stats


def step_collect_news(db_path, tickers, start_date, end_date):
    if not FINNHUB_API_KEY:
        print("   ⚠ Skipped (set NEWS_API_KEY in .env for news)")
        return None
    stats = NewsCollector(FINNHUB_API_KEY, db_path).collect(
        tickers, start_date, end_date
    )
    print(f"   ✓ {stats['rows_added']} articles "
          f"({stats['duplicates_skipped']} dupes skipped)")
    return stats


def step_validate(db_path, tickers):
    results = PriceDataValidator(db_path).validate(tickers)
    for info in results["coverage"]:
        print(f"   {info['ticker']:5s}  {info['days_collected']} trading days  "
              f"({info['first_date']} → {info['last_date']})")
    issues = (len(results["missing_data"])
              + len(results["price_anomalies"])
              + len(results["volume_anomalies"]))
    if issues:
        print(f"   ⚠ {issues} data quality issues found")
    else:
        print("   ✓ No data quality issues")


def step_labels(tickers):
    summary = LabelGenerator().generate(tickers)
    print(f"   ✓ {summary['total_labels']} labels "
          f"({summary['total_skipped']} already existed)")
    return summary["total_labels"]


def step_split():
    summary = DatasetSplitter().split()
    for name in ["train", "val", "test"]:
        s = summary[name]
        print(f"   {name:5s}  {s['num_days']:3d} days  {s['date_range']}  "
              f"({s['prices']} prices, {s['labels']} labels)")


def step_build_features(tickers):
    gen = SequenceGenerator()
    all_X, all_y, all_dates = [], [], []

    for ticker in tickers:
        X, y, dates = gen.generate(ticker)
        if len(X) == 0:
            print(f"   {ticker:5s}  skipped (need ≥{gen.seq_len + 1} valid rows "
                  f"after indicator warmup)")
            continue
        print(f"   {ticker:5s}  {len(X)} sequences, {y.mean():.0%} up")
        all_X.append(X)
        all_y.append(y)
        all_dates.extend([(ticker, d) for d in dates])

    if not all_X:
        print("   ✗ No sequences — increase --days (need ≥250 for LSTM)")
        return False

    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / "X_sequences.npy", X_combined)
    np.save(PROCESSED_DATA_DIR / "y_labels.npy", y_combined)
    with open(PROCESSED_DATA_DIR / "sequence_dates.json", "w") as f:
        json.dump(all_dates, f)

    print(f"\n   ✓ {len(X_combined)} total sequences, shape {X_combined.shape}")
    return True


def step_train_lstm():
    from src.models.lstm_model import StockLSTM, LSTMTrainer

    X = np.load(PROCESSED_DATA_DIR / "X_sequences.npy")
    y = np.load(PROCESSED_DATA_DIR / "y_labels.npy")
    with open(PROCESSED_DATA_DIR / "sequence_dates.json") as f:
        dates_meta = json.load(f)
    with open(PROCESSED_DATA_DIR / "split_info.json") as f:
        split_info = json.load(f)

    train_dates = set(split_info["splits"]["train"]["dates"])
    val_dates = set(split_info["splits"]["val"]["dates"])
    test_dates = set(split_info["splits"]["test"]["dates"])

    train_idx, val_idx, test_idx = [], [], []
    for i, (_, date) in enumerate(dates_meta):
        if date in train_dates:
            train_idx.append(i)
        elif date in val_dates:
            val_idx.append(i)
        elif date in test_dates:
            test_idx.append(i)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"   Split → Train: {len(X_train)}  Val: {len(X_val)}  "
          f"Test: {len(X_test)}")

    if len(X_train) == 0:
        print("   ✗ No training data — skipping LSTM")
        return None

    config = {**LSTM_CONFIG, "epochs": 50}
    model = StockLSTM(input_size=X_train.shape[2],
                      hidden_sizes=config["lstm_units"],
                      dropout=config["dropout"])
    trainer = LSTMTrainer(model, config)
    trainer.train(X_train, y_train, X_val, y_val, patience=15)
    test_acc = trainer.evaluate(X_test, y_test, split_name="Test")
    trainer.save(MODELS_DIR / "lstm_model.pt")
    return test_acc


def step_train_nlp(tickers):
    from src.models.nlp_baseline import NLPBaseline

    model = NLPBaseline()
    texts, labels, metadata = model.extractor.prepare(tickers)

    if not texts:
        print("   ⚠ No news data — skipping NLP baseline")
        return None

    with open(PROCESSED_DATA_DIR / "split_info.json") as f:
        split_info = json.load(f)

    train_dates = set(split_info["splits"]["train"]["dates"])
    val_dates = set(split_info["splits"]["val"]["dates"])

    train_idx, val_idx, test_idx = [], [], []
    for i, (_, date) in enumerate(metadata):
        if date in train_dates:
            train_idx.append(i)
        elif date in val_dates:
            val_idx.append(i)
        else:
            test_idx.append(i)

    train_texts = [texts[i] for i in train_idx]
    y_train = labels[train_idx]

    if not train_texts:
        print("   ⚠ No training samples — Finnhub free tier only returns "
              "recent articles (~60 days), which fall outside the training "
              "date range. Collect news over time to build up training data.")
        return None

    print(f"   {len(texts)} samples with news → "
          f"Train: {len(train_texts)}  Val: {len(val_idx)}  "
          f"Test: {len(test_idx)}")

    X_train = model.extractor.fit_transform(train_texts)
    model.train(X_train, y_train)

    test_acc = None
    if test_idx:
        test_texts = [texts[i] for i in test_idx]
        X_test = model.extractor.transform(test_texts)
        y_test = labels[test_idx]
        test_acc = model.evaluate(X_test, y_test, split_name="Test")

    model.save(MODELS_DIR / "nlp_baseline.joblib")
    return test_acc


def step_sample_data(db_path, tickers):
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
        print(f"\n   Sample headlines:")
        for r in rows:
            title = r[1][:65] + "..." if len(r[1]) > 65 else r[1]
            print(f"     [{r[0]}] {title}")

    rows = conn.execute(
        f"SELECT ticker, date, label_binary, ROUND(label_return, 2) "
        f"FROM labels WHERE ticker IN ({ph}) ORDER BY date DESC LIMIT 5",
        tickers,
    ).fetchall()
    if rows:
        print(f"\n   Recent labels:")
        for r in rows:
            direction = "UP" if r[2] == 1 else "DN"
            print(f"     {r[0]:5s} {r[1]}  {direction}  {r[3]:+.2f}%")

    conn.close()


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full end-to-end pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/demo.py                        # all 15 tickers, 250 days
  python scripts/demo.py --reset                # fresh start
  python scripts/demo.py --tickers AAPL TSLA    # quick test (fewer tickers)
  python scripts/demo.py --days 365             # more history
  python scripts/demo.py --skip-training        # data pipeline only""",
    )
    parser.add_argument("--reset", action="store_true",
                        help="Wipe all data before running")
    parser.add_argument("--skip-training", action="store_true",
                        help="Data pipeline only, skip model training")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help=f"Tickers to use (default: all {len(TICKERS)})")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with just AAPL + TSLA (fewer samples)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Calendar days of price history (default: {DEFAULT_DAYS})")
    args = parser.parse_args()

    if args.quick:
        tickers = ["AAPL", "TSLA"]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = DEFAULT_TICKERS

    days = args.days

    print("=" * 70)
    print("NEWS-TO-ALPHA  —  FULL PIPELINE DEMO")
    print("=" * 70)
    print(f"Tickers : {', '.join(tickers)}  ({len(tickers)} total)")
    print(f"History : {days} calendar days")

    if args.reset:
        step_reset()

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"\n1. Setting up database...")
    db_path = step_database()

    print(f"\n2. Collecting prices ({start_date} → {end_date})...")
    step_collect_prices(db_path, tickers, start_date, end_date)

    print(f"\n3. Collecting news...")
    step_collect_news(db_path, tickers, start_date, end_date)

    print(f"\n4. Validating data...")
    step_validate(db_path, tickers)

    print(f"\n5. Generating labels...")
    label_count = step_labels(tickers)
    if label_count == 0:
        print("\n   ✗ No labels — check price data.")
        return

    print(f"\n6. Splitting dataset (70 / 15 / 15)...")
    step_split()

    print(f"\n7. Building features (indicators → {LSTM_CONFIG['sequence_length']}-day "
          f"LSTM sequences)...")
    has_sequences = step_build_features(tickers)

    print(f"\n8. Sample data...")
    step_sample_data(db_path, tickers)

    lstm_acc, nlp_acc = None, None
    if not args.skip_training and has_sequences:
        print(f"\n9.  Training LSTM (up to 50 epochs)...")
        lstm_acc = step_train_lstm()

        print(f"\n10. Training NLP baseline...")
        nlp_acc = step_train_nlp(tickers)

    # ── summary ──
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"\n  Database : data/database.db")
    print(f"  Features : data/processed/")
    print(f"  Models   : data/models/")
    if lstm_acc is not None:
        print(f"\n  LSTM test accuracy : {lstm_acc:.4f}")
    if nlp_acc is not None:
        print(f"  NLP  test accuracy : {nlp_acc:.4f}")
    print(f"  Random baseline    : 0.5000")

    print(f"\nTo improve results:")
    print(f"  python scripts/demo.py --days 365           # more history")
    print(f"  python scripts/train_lstm.py --epochs 100   # longer training")
    print()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
