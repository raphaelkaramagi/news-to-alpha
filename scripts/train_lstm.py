#!/usr/bin/env python3
"""
Train the 2-layer LSTM on price + indicator sequences.

Loads the numpy arrays built by build_features.py, splits them
chronologically using split_info.json, and trains the LSTM.

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --epochs 100 --batch-size 64
    python scripts/train_lstm.py --lr 0.0005
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, LSTM_CONFIG  # noqa: E402
from src.models.lstm_model import StockLSTM, LSTMTrainer  # noqa: E402


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


def split_by_dates(X: np.ndarray, y: np.ndarray,
                   dates_meta: list) -> tuple:
    """Assign each sequence to train/val/test based on its date."""
    split_path = PROCESSED_DATA_DIR / "split_info.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found — run: python scripts/split_dataset.py"
        )
    with open(split_path) as f:
        split_info = json.load(f)

    train_dates = set(split_info["splits"]["train"]["dates"])
    val_dates = set(split_info["splits"]["val"]["dates"])
    test_dates = set(split_info["splits"]["test"]["dates"])

    train_idx, val_idx, test_idx = [], [], []
    for i, (ticker, date) in enumerate(dates_meta):
        if date in train_dates:
            train_idx.append(i)
        elif date in val_dates:
            val_idx.append(i)
        elif date in test_dates:
            test_idx.append(i)

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Training epochs (default: {LSTM_CONFIG['epochs']})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {LSTM_CONFIG['batch_size']})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {LSTM_CONFIG['learning_rate']})")
    args = parser.parse_args()

    print("=" * 60)
    print("LSTM TRAINING")
    print("=" * 60)

    # ---- Load data ----
    print("\n--- Loading data ---")
    X, y, dates_meta = load_data()
    print(f"  Total sequences: {len(X)}, shape: {X.shape}")

    # ---- Split by date ----
    print("\n--- Splitting by date ---")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_dates(
        X, y, dates_meta
    )
    print(f"  Train: {len(X_train)} ({y_train.mean():.0%} up)")
    print(f"  Val:   {len(X_val)} ({y_val.mean():.0%} up)")
    print(f"  Test:  {len(X_test)} ({y_test.mean():.0%} up)")

    if len(X_train) == 0:
        print("\nNo training data — collect more prices "
              "(python scripts/collect_prices.py --days 90).")
        return

    # ---- Build config with CLI overrides ----
    config = {**LSTM_CONFIG}
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr

    # ---- Create model ----
    input_size = X_train.shape[2]
    model = StockLSTM(
        input_size=input_size,
        hidden_sizes=config["lstm_units"],
        dropout=config["dropout"],
    )

    print(f"\n  Model: 2-layer LSTM ({config['lstm_units']}), "
          f"dropout={config['dropout']}, lr={config['learning_rate']}")
    print(f"  Input: ({config['sequence_length']} timesteps, {input_size} features)")

    # ---- Train ----
    print("\n--- Training ---")
    trainer = LSTMTrainer(model, config)
    trainer.train(X_train, y_train, X_val, y_val)

    # ---- Evaluate on test set ----
    print("\n--- Evaluation ---")
    test_acc = trainer.evaluate(X_test, y_test, split_name="Test")

    # ---- Save ----
    save_path = MODELS_DIR / "lstm_model.pt"
    trainer.save(save_path)

    print(f"\n{'=' * 60}")
    print(f"  Model saved to {save_path}")
    print(f"  Test accuracy:    {test_acc:.4f}")
    print(f"  Random baseline:  0.5000")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    main()
