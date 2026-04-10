#!/usr/bin/env python3
"""
Train the 2-layer LSTM on price + indicator sequences and export predictions.

Outputs
-------
  data/models/lstm_model.pt                  – trained model checkpoint
  data/processed/price_predictions.csv       – per ticker-day predictions

The CSV follows the session 1 contract: rows keyed by (ticker, prediction_date)
with columns financial_pred_proba, financial_pred_binary, financial_confidence,
actual_binary, model_name, model_version.

Usage
-----
  python scripts/train_lstm.py
  python scripts/train_lstm.py --epochs 100 --batch-size 64
  python scripts/train_lstm.py --lr 0.0005
  python scripts/train_lstm.py --no-db-export
"""

import sys
import json
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, LSTM_CONFIG, DATABASE_PATH  # noqa: E402
from src.models.lstm_model import StockLSTM, LSTMTrainer  # noqa: E402

MODEL_NAME = "lstm_price"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
CSV_NAME = "price_predictions.csv"

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Load sequences, labels, and date metadata from the processed dir."""
    X = np.load(PROCESSED_DATA_DIR / "X_sequences.npy")
    y = np.load(PROCESSED_DATA_DIR / "y_labels.npy")

    dates_path = PROCESSED_DATA_DIR / "sequence_dates.json"
    if not dates_path.exists():
        raise FileNotFoundError(
            f"{dates_path} not found — re-run: python scripts/build_features.py"
        )
    with open(dates_path) as f:
        dates_meta = json.load(f)

    return X, y, dates_meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_by_dates(X: np.ndarray, y: np.ndarray,
                   dates_meta: list) -> dict[str, tuple[np.ndarray, np.ndarray, list]]:
    """
    Assign each sequence to train/val/test based on its date.

    Returns a dict of {split_name: (X_split, y_split, meta_split)} where
    meta_split is the list of [ticker, date] pairs for that split.
    """
    split_path = PROCESSED_DATA_DIR / "split_info.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found — run: python scripts/split_dataset.py"
        )
    with open(split_path) as f:
        split_info = json.load(f)

    date_sets = {
        name: set(split_info["splits"][name]["dates"])
        for name in ("train", "val", "test")
    }

    indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, (ticker, date) in enumerate(dates_meta):
        for name, dset in date_sets.items():
            if date in dset:
                indices[name].append(i)
                break

    result = {}
    for name, idx in indices.items():
        result[name] = (
            X[idx],
            y[idx],
            [dates_meta[i] for i in idx],
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREDICTION EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_export(
    trainer: LSTMTrainer,
    split_name: str,
    X_split: np.ndarray,
    y_split: np.ndarray,
    dates_split: list[list[str]],
) -> pd.DataFrame:
    """
    Build a prediction DataFrame for one split.

    Each row corresponds to one sequence and contains the model's probability,
    binary prediction, confidence, and the ground-truth label — keyed by
    (ticker, prediction_date) per the session 1 contract.
    """
    proba = trainer.predict_proba(X_split)
    binary = (proba >= 0.5).astype(int)
    confidence = np.abs(proba - 0.5) * 2

    tickers = [entry[0] for entry in dates_split]
    dates = [entry[1] for entry in dates_split]

    return pd.DataFrame({
        "ticker": tickers,
        "prediction_date": dates,
        "split": split_name,
        "financial_pred_proba": proba,
        "financial_pred_binary": binary,
        "financial_confidence": confidence,
        "actual_binary": y_split.astype(int),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    })


def save_predictions_csv(
    trainer: LSTMTrainer,
    splits: dict[str, tuple[np.ndarray, np.ndarray, list]],
    out_path: Path,
) -> Path:
    """Export predictions for all splits to a single CSV."""
    frames = []
    for split_name, (X_s, y_s, meta_s) in splits.items():
        if len(X_s) == 0:
            continue
        frames.append(build_prediction_export(trainer, split_name, X_s, y_s, meta_s))

    if not frames:
        raise RuntimeError("No prediction rows generated — all splits are empty.")

    combined = pd.concat(frames, ignore_index=True)

    col_order = [
        "ticker", "prediction_date", "split",
        "financial_pred_proba", "financial_pred_binary", "financial_confidence",
        "actual_binary", "model_name", "model_version",
    ]
    combined = combined[col_order]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    log.info("Predictions saved to %s  (%d rows)", out_path, len(combined))
    return out_path


def upsert_predictions_db(
    trainer: LSTMTrainer,
    splits: dict[str, tuple[np.ndarray, np.ndarray, list]],
    db_path: str | Path,
) -> int:
    """
    Insert (or replace) rows into the predictions table.
    Only fills financial_* columns; news_* and ensemble_* remain NULL.
    """
    frames = []
    for split_name, (X_s, y_s, meta_s) in splits.items():
        if len(X_s) == 0:
            continue
        frames.append(build_prediction_export(trainer, split_name, X_s, y_s, meta_s))

    if not frames:
        return 0

    combined = pd.concat(frames, ignore_index=True)

    conn = sqlite3.connect(str(db_path))
    count = 0
    try:
        for _, row in combined.iterrows():
            conn.execute(
                """INSERT INTO predictions
                       (ticker, date, financial_pred_proba, financial_confidence,
                        financial_pred_binary, actual_binary, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, date, model_version) DO UPDATE SET
                       financial_pred_proba  = excluded.financial_pred_proba,
                       financial_confidence  = excluded.financial_confidence,
                       financial_pred_binary = excluded.financial_pred_binary,
                       actual_binary         = excluded.actual_binary
                """,
                (
                    row["ticker"],
                    row["prediction_date"],
                    float(row["financial_pred_proba"]),
                    float(row["financial_confidence"]),
                    int(row["financial_pred_binary"]),
                    int(row["actual_binary"]),
                    row["model_version"],
                ),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()

    log.info("Upserted %d rows into predictions table.", count)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model and export predictions")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Training epochs (default: {LSTM_CONFIG['epochs']})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {LSTM_CONFIG['batch_size']})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {LSTM_CONFIG['learning_rate']})")
    parser.add_argument("--no-db-export", action="store_true",
                        help="Skip writing predictions to the database predictions table")
    parser.add_argument("--output", default=None,
                        help="Override CSV output path")
    args = parser.parse_args()

    csv_path = Path(args.output) if args.output else (PROCESSED_DATA_DIR / CSV_NAME)
    model_path = MODELS_DIR / "lstm_model.pt"

    print("=" * 70)
    print("LSTM TRAINING + EXPORT")
    print("=" * 70)
    print(f"  Model output  : {model_path}")
    print(f"  CSV output    : {csv_path}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n--- Loading data ---")
    X, y, dates_meta = load_data()
    print(f"  Total sequences: {len(X)}, shape: {X.shape}")

    # ── 2. Split by date ─────────────────────────────────────────────────────
    print("\n--- Splitting by date ---")
    splits = split_by_dates(X, y, dates_meta)
    for name in ("train", "val", "test"):
        X_s, y_s, _ = splits[name]
        pct_up = y_s.mean() if len(y_s) > 0 else 0
        print(f"  {name.capitalize():5s}: {len(X_s)} ({pct_up:.0%} up)")

    X_train, y_train, _ = splits["train"]
    if len(X_train) == 0:
        print("\nNo training data — collect more prices "
              "(python scripts/collect_prices.py --days 90).")
        return

    # ── 3. Build config with CLI overrides ───────────────────────────────────
    config = {**LSTM_CONFIG}
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr

    # ── 4. Create model ──────────────────────────────────────────────────────
    input_size = X_train.shape[2]
    model = StockLSTM(
        input_size=input_size,
        hidden_sizes=config["lstm_units"],
        dropout=config["dropout"],
    )

    print(f"\n  Model: 2-layer LSTM ({config['lstm_units']}), "
          f"dropout={config['dropout']}, lr={config['learning_rate']}")
    print(f"  Input: ({config['sequence_length']} timesteps, {input_size} features)")

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print("\n--- Training ---")
    trainer = LSTMTrainer(model, config)
    X_val, y_val, _ = splits["val"]
    trainer.train(X_train, y_train, X_val, y_val)

    # ── 6. Evaluate on test set ──────────────────────────────────────────────
    print("\n--- Evaluation ---")
    X_test, y_test, _ = splits["test"]
    test_acc = trainer.evaluate(X_test, y_test, split_name="Test")

    # ── 7. Save model ────────────────────────────────────────────────────────
    trainer.save(model_path)
    print(f"\n  Model saved: {model_path}")

    # ── 8. Export predictions CSV ────────────────────────────────────────────
    print("\n--- Exporting predictions ---")
    save_predictions_csv(trainer, splits, csv_path)
    print(f"  CSV saved:   {csv_path}")

    # ── 9. Optional DB upsert ────────────────────────────────────────────────
    if not args.no_db_export:
        n = upsert_predictions_db(trainer, splits, DATABASE_PATH)
        print(f"  DB rows:     {n} upserted into predictions table")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Test accuracy    : {test_acc:.4f}")
    print(f"  Random baseline  : 0.5000")
    print(f"  model_name       : {MODEL_NAME}")
    print(f"  model_version    : {MODEL_VERSION}")
    print(f"  model_file       : {model_path}")
    print(f"  predictions      : {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    main()
