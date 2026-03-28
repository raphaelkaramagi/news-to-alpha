#!/usr/bin/env python3
"""
Train the NLP baseline (logistic regression on TF-IDF headlines).

Extracts TF-IDF features from news headlines, splits chronologically,
and trains a logistic-regression classifier.

Usage:
    python scripts/train_nlp.py
    python scripts/train_nlp.py --tickers AAPL TSLA
    python scripts/train_nlp.py --max-features 3000
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TICKERS, PROCESSED_DATA_DIR, MODELS_DIR, NLP_CONFIG  # noqa: E402
from src.models.nlp_baseline import NLPBaseline  # noqa: E402


def split_by_dates(metadata, texts, labels):
    """Split samples into train/val/test based on their dates."""
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
    for i, (ticker, date) in enumerate(metadata):
        if date in train_dates:
            train_idx.append(i)
        elif date in val_dates:
            val_idx.append(i)
        elif date in test_dates:
            test_idx.append(i)

    def _select(indices):
        t = [texts[i] for i in indices]
        la = labels[indices]
        return t, la

    return _select(train_idx), _select(val_idx), _select(test_idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NLP baseline")
    parser.add_argument("--tickers", nargs="+",
                        help="Specific tickers (default: all)")
    parser.add_argument("--max-features", type=int, default=None,
                        help=f"TF-IDF vocabulary size (default: {NLP_CONFIG['max_features']})")
    args = parser.parse_args()

    tickers = args.tickers or TICKERS
    max_features = args.max_features or NLP_CONFIG["max_features"]

    print("=" * 60)
    print("NLP BASELINE TRAINING")
    print("=" * 60)

    # ---- Prepare text data ----
    print("\n--- Preparing text features ---")
    model = NLPBaseline(max_features=max_features)
    texts, labels, metadata = model.extractor.prepare(tickers)

    if not texts:
        print("\nNo samples with news coverage — collect news first "
              "(python scripts/collect_news.py --tickers AAPL --days 30).")
        return

    print(f"  Samples with news: {len(texts)}")
    print(f"  Label distribution: {labels.mean():.0%} up / "
          f"{1 - labels.mean():.0%} down")

    # ---- Split ----
    print("\n--- Splitting by date ---")
    (train_texts, y_train), (val_texts, y_val), (test_texts, y_test) = \
        split_by_dates(metadata, texts, labels)

    print(f"  Train: {len(train_texts)}")
    print(f"  Val:   {len(val_texts)}")
    print(f"  Test:  {len(test_texts)}")

    if not train_texts:
        print("\nNo training samples — need more news data.")
        return

    # ---- Fit TF-IDF on training data only, then transform all splits ----
    print("\n--- Fitting TF-IDF ---")
    X_train = model.extractor.fit_transform(train_texts)
    X_val = model.extractor.transform(val_texts) if val_texts else None
    X_test = model.extractor.transform(test_texts) if test_texts else None

    print(f"  Vocabulary size: {X_train.shape[1]}")

    # ---- Train ----
    print("\n--- Training logistic regression ---")
    model.train(X_train, y_train)

    # ---- Evaluate ----
    print("\n--- Evaluation ---")
    model.evaluate(X_train, y_train, split_name="Train")

    val_acc = None
    if X_val is not None and len(y_val) > 0:
        val_acc = model.evaluate(X_val, y_val, split_name="Val")

    test_acc = None
    if X_test is not None and len(y_test) > 0:
        test_acc = model.evaluate(X_test, y_test, split_name="Test")

    # ---- Save ----
    save_path = MODELS_DIR / "nlp_baseline.joblib"
    model.save(save_path)

    print(f"\n{'=' * 60}")
    print(f"  Model saved to {save_path}")
    if test_acc is not None:
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
