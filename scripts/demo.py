#!/usr/bin/env python3
"""
Full end-to-end demo: collect data → build features → train models → export predictions.

The single command to verify the entire pipeline works.  Collects enough
price history for the 60-day LSTM windows (~250 calendar days = ~175
trading days, minus ~34 for indicator warmup = ~140 usable rows).

Usage:
    python scripts/demo.py                          # all 15 tickers, 250 days
    python scripts/demo.py --reset                  # wipe first, then run
    python scripts/demo.py --quick                  # fast test (AAPL + TSLA)
    python scripts/demo.py --tickers AAPL NVDA TSLA # pick your own
    python scripts/demo.py --days 365               # more history
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
    print("\n0. Resetting features + models (keeping database)...")
    reset(keep_db=True)
    print("   ✓ Clean slate (news articles preserved)")


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
    total = summary['total_labels'] + summary['total_skipped']
    if summary['total_labels'] > 0:
        print(f"   ✓ {summary['total_labels']} new labels "
              f"({summary['total_skipped']} already existed)")
    else:
        print(f"   ✓ {total} labels (all already existed)")


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
    from scripts.train_lstm import load_data, split_by_dates, \
        save_predictions_csv, MODEL_NAME
    from src.models.lstm_model import StockLSTM, LSTMTrainer

    X, y, dates_meta = load_data()
    splits = split_by_dates(X, y, dates_meta)

    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, _ = splits["test"]

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

    csv_path = PROCESSED_DATA_DIR / "price_predictions.csv"
    save_predictions_csv(trainer, splits, csv_path)
    print(f"   ✓ Predictions saved to {csv_path.name}")

    return test_acc


def step_train_nlp(tickers):
    """Run Moses's cutoff-aligned TF-IDF pipeline (scripts/train_nlp.py)."""
    from scripts.train_nlp import (
        build_dataset, chronological_split, build_pipeline,
        train as train_nlp_model, evaluate as eval_nlp,
        save_predictions_csv as save_nlp_csv,
    )
    import joblib

    try:
        df = build_dataset(DATABASE_PATH)
    except RuntimeError as e:
        print(f"   ⚠ {e}")
        return None

    if df.empty:
        print("   ⚠ No news data — skipping NLP baseline")
        return None

    try:
        train_df, val_df, test_df = chronological_split(df)
    except ValueError as e:
        print(f"   ⚠ {e}")
        return None

    splits = {"train": train_df, "val": val_df, "test": test_df}
    print(f"   {len(df)} ticker-day rows → "
          f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    pipe = train_nlp_model(train_df)

    test_metrics = eval_nlp(pipe, test_df, "test")
    test_acc = test_metrics.get("accuracy")

    model_path = MODELS_DIR / "nlp_baseline.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    csv_path = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"
    save_nlp_csv(pipe, splits, csv_path)
    print(f"   ✓ Predictions saved to {csv_path.name}")

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
    step_labels(tickers)

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
    print(f"\n  Database    : data/database.db")
    print(f"  Features    : data/processed/")
    print(f"  Models      : data/models/")
    if lstm_acc is not None:
        print(f"\n  LSTM test accuracy : {lstm_acc:.4f}")
        print(f"    → data/processed/price_predictions.csv")
    if nlp_acc is not None:
        print(f"  NLP  test accuracy : {nlp_acc:.4f}")
        print(f"    → data/processed/news_tfidf_predictions.csv")
    print(f"  Random baseline    : 0.5000")

    print(f"\nNext steps:")
    print(f"  python scripts/train_lstm.py --epochs 100        # longer training")
    print(f"  python scripts/train_news_embeddings.py          # embeddings model")
    print(f"  python scripts/demo.py --days 365                # more history")
    print()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
