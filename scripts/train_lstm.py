#!/usr/bin/env python3
"""Train the 2-layer LSTM (with ticker embedding) and export predictions.

This script now owns BOTH feature assembly and training so that the
StandardScaler fit on the train split is always in lock-step with the
model.  The old flow of building .npy files with per-window min-max
normalization has been removed - sequences are raw, scaling happens here
exactly once, and the scaler state is saved inside the checkpoint.

Outputs
-------
  data/models/lstm_model.pt                     - model weights + scaler + ticker_to_idx
  data/processed/price_predictions.csv          - per ticker-day predictions
  data/processed/X_sequences.npy (optional)     - raw sequences used (for debugging)

Usage
-----
  python scripts/train_lstm.py
  python scripts/train_lstm.py --epochs 100 --batch-size 64
  python scripts/train_lstm.py --lr 0.0005
  python scripts/train_lstm.py --no-db-export
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    LSTM_CONFIG,
    DATABASE_PATH,
    TICKERS,
    RANDOM_SEED,
)
from src.features.sequence_generator import (  # noqa: E402
    SequenceGenerator,
    FEATURE_COLUMNS,
    LEVEL_IDX,
)
from src.models.lstm_model import StockLSTM, LSTMTrainer  # noqa: E402

MODEL_NAME = "lstm_price"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
CSV_NAME = "price_predictions.csv"

log = logging.getLogger(__name__)


def _seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_sequences_for_all_tickers(
    tickers: list[str],
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]], dict[str, int]]:
    """Generate raw (unscaled) sequences for every ticker, tagged with a ticker index.

    Returns:
        X           : (N, seq_len, num_features) float32
        y           : (N,) int32
        returns     : (N,) float32 pct-return of the target horizon
        ticker_idx  : (N,) int64
        meta        : list of (ticker, prediction_date) tuples, length N
        ticker_to_idx : map ticker -> int ID used by the embedding
    """
    ticker_to_idx = {t: i for i, t in enumerate(sorted(set(tickers)))}

    gen = SequenceGenerator(horizon=horizon)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    ret_list: list[np.ndarray] = []
    tidx_list: list[np.ndarray] = []
    meta: list[tuple[str, str]] = []

    for ticker in tickers:
        X, y, returns, dates = gen.generate(ticker)
        if len(X) == 0:
            print(f"  {ticker:5s}  skipped (not enough data)")
            continue
        print(f"  {ticker:5s}  {len(X)} sequences")
        X_list.append(X)
        y_list.append(y)
        ret_list.append(returns)
        tidx_list.append(
            np.full(len(X), ticker_to_idx[ticker], dtype=np.int64)
        )
        meta.extend((ticker, d) for d in dates)

    if not X_list:
        raise RuntimeError(
            "No sequences built - run `python scripts/demo.py` or "
            "`python scripts/collect_prices.py` to ingest more data."
        )

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    ret_all = np.concatenate(ret_list, axis=0)
    tidx_all = np.concatenate(tidx_list, axis=0)
    return X_all, y_all, ret_all, tidx_all, meta, ticker_to_idx


def split_by_dates(
    X: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    tidx: np.ndarray,
    meta: list[tuple[str, str]],
) -> dict[str, dict]:
    split_path = PROCESSED_DATA_DIR / "split_info.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found - run `python scripts/split_dataset.py` first."
        )
    with open(split_path) as f:
        split_info = json.load(f)

    date_sets = {
        name: set(split_info["splits"][name]["dates"])
        for name in ("train", "val", "test")
    }

    indices = {"train": [], "val": [], "test": []}
    for i, (_, date) in enumerate(meta):
        for name, dset in date_sets.items():
            if date in dset:
                indices[name].append(i)
                break

    out = {}
    for name, idx in indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        out[name] = {
            "X": X[idx_arr],
            "y": y[idx_arr],
            "returns": returns[idx_arr],
            "tidx": tidx[idx_arr],
            "meta": [meta[i] for i in idx],
        }
    return out


def fit_level_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit mean / std on the level-feature slice of training windows.

    We flatten windows across the time axis so each level feature has
    (N_train * seq_len) observations contributing to its statistics.
    """
    level = X_train[:, :, LEVEL_IDX]
    flat = level.reshape(-1, level.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_level_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = X.copy()
    out[:, :, LEVEL_IDX] = (out[:, :, LEVEL_IDX] - mean) / std
    return out.astype(np.float32)


class SeedEnsemble:
    """Wraps N trained LSTMTrainers and averages their calibrated probabilities."""

    def __init__(self, trainers: list[LSTMTrainer]):
        if not trainers:
            raise ValueError("SeedEnsemble requires at least one trainer")
        self.trainers = trainers

    def predict_proba(self, X: np.ndarray,
                      ticker_idx: np.ndarray | None = None) -> np.ndarray:
        parts = [t.predict_proba(X, ticker_idx) for t in self.trainers]
        return np.mean(np.stack(parts, axis=0), axis=0)

    def predict(self, X: np.ndarray,
                ticker_idx: np.ndarray | None = None) -> np.ndarray:
        return (self.predict_proba(X, ticker_idx) >= 0.5).astype(np.int32)


def _prob_predict(predictor, X, tidx) -> np.ndarray:
    """Handle both LSTMTrainer and SeedEnsemble transparently."""
    return predictor.predict_proba(X, tidx)


def build_prediction_export(
    predictor,
    split_name: str,
    X_split: np.ndarray,
    y_split: np.ndarray,
    tidx_split: np.ndarray,
    meta_split: list[tuple[str, str]],
) -> pd.DataFrame:
    if len(X_split) == 0:
        return pd.DataFrame()
    proba = _prob_predict(predictor, X_split, tidx_split)
    binary = (proba >= 0.5).astype(int)
    confidence = np.abs(proba - 0.5) * 2

    return pd.DataFrame({
        "ticker": [m[0] for m in meta_split],
        "prediction_date": [m[1] for m in meta_split],
        "split": split_name,
        "financial_pred_proba": proba,
        "financial_pred_binary": binary,
        "financial_confidence": confidence,
        "actual_binary": y_split.astype(int),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    })


def save_predictions_csv(
    predictor,
    splits: dict[str, dict],
    out_path: Path,
) -> Path:
    frames = []
    for split_name, s in splits.items():
        df = build_prediction_export(
            predictor, split_name, s["X"], s["y"], s["tidx"], s["meta"]
        )
        if not df.empty:
            frames.append(df)
    if not frames:
        raise RuntimeError("No prediction rows generated - all splits are empty.")

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
    predictor,
    splits: dict[str, dict],
    db_path: str | Path,
) -> int:
    frames = []
    for split_name, s in splits.items():
        df = build_prediction_export(
            predictor, split_name, s["X"], s["y"], s["tidx"], s["meta"]
        )
        if not df.empty:
            frames.append(df)
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
                    row["ticker"], row["prediction_date"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model and export predictions")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to train on (default: all configured TICKERS)")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3],
                        help="Prediction horizon in trading days (default: 1)")
    parser.add_argument("--min-move-pct", type=float, default=0.0,
                        help="Drop TRAIN rows where |return_horizon| < this (eval keeps all).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Train multiple LSTMs with these seeds and average their probs "
                             "(e.g. --seeds 42 1337 2024). Default: single-seed from --seed.")
    parser.add_argument("--no-db-export", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    _seed_everything(args.seed)

    csv_path = Path(args.output) if args.output else (PROCESSED_DATA_DIR / CSV_NAME)
    model_path = MODELS_DIR / "lstm_model.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = [t.upper() for t in args.tickers] if args.tickers else list(TICKERS)

    print("=" * 70)
    print("LSTM TRAINING + EXPORT")
    print("=" * 70)
    print(f"  Tickers      : {', '.join(tickers)}")
    print(f"  Model output : {model_path}")
    print(f"  CSV output   : {csv_path}")

    print("\n--- Building sequences per ticker ---")
    X_all, y_all, ret_all, tidx_all, meta, ticker_to_idx = build_sequences_for_all_tickers(
        tickers, horizon=args.horizon,
    )
    print(f"  Total sequences: {len(X_all)}, shape: {X_all.shape}, "
          f"features: {len(FEATURE_COLUMNS)}  (horizon={args.horizon})")

    print("\n--- Splitting by date ---")
    splits = split_by_dates(X_all, y_all, ret_all, tidx_all, meta)
    for name in ("train", "val", "test"):
        s = splits[name]
        pct = s["y"].mean() if len(s["y"]) else 0.0
        print(f"  {name.capitalize():5s}: {len(s['X'])} ({pct:.0%} up)")

    if args.min_move_pct > 0 and len(splits["train"]["X"]) > 0:
        keep = np.abs(splits["train"]["returns"]) >= args.min_move_pct
        dropped = int((~keep).sum())
        splits["train"] = {
            "X": splits["train"]["X"][keep],
            "y": splits["train"]["y"][keep],
            "returns": splits["train"]["returns"][keep],
            "tidx": splits["train"]["tidx"][keep],
            "meta": [m for m, k in zip(splits["train"]["meta"], keep) if k],
        }
        print(f"  Drop-flat filter (|ret|<{args.min_move_pct}%): "
              f"dropped {dropped} train rows, kept {len(splits['train']['X'])}")

    if len(splits["train"]["X"]) == 0:
        print("\nNo training data - run `python scripts/demo.py` to ingest data.")
        return

    print("\n--- Fitting level scaler on training data ---")
    mean, std = fit_level_scaler(splits["train"]["X"])
    for name in ("train", "val", "test"):
        splits[name]["X"] = apply_level_scaler(splits[name]["X"], mean, std)
    print(f"  Scaled {len(LEVEL_IDX)} level features; "
          f"{len(FEATURE_COLUMNS) - len(LEVEL_IDX)} kept scale-invariant.")

    config = {**LSTM_CONFIG}
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.dropout is not None:
        config["dropout"] = args.dropout

    input_size = X_all.shape[2]
    scaler_state = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "level_idx": LEVEL_IDX,
        "feature_columns": FEATURE_COLUMNS,
    }

    y_train = splits["train"]["y"].astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        pos_weight = 1.0
    else:
        pos_weight = n_neg / n_pos
    print(f"\n  Class balance train: pos={n_pos} neg={n_neg}  pos_weight={pos_weight:.3f}")

    seeds = args.seeds or config.get("seeds") or [args.seed]
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    print(f"\n--- Training seed ensemble ({len(seeds)} models: seeds={seeds}) ---")

    trainers: list[LSTMTrainer] = []
    patience = int(config.get("patience", 10))

    for i, seed in enumerate(seeds, start=1):
        print(f"\n>> Seed {i}/{len(seeds)}: {seed}")
        _seed_everything(seed)
        model_i = StockLSTM(
            input_size=input_size,
            hidden_sizes=config["lstm_units"],
            dropout=config["dropout"],
            num_tickers=len(ticker_to_idx),
            ticker_embed_dim=config.get("ticker_embed_dim", 4),
        )
        trainer_i = LSTMTrainer(
            model_i, config,
            ticker_to_idx=ticker_to_idx,
            scaler_state=scaler_state,
            pos_weight=pos_weight,
        )
        trainer_i.train(
            splits["train"]["X"], splits["train"]["y"],
            splits["val"]["X"], splits["val"]["y"],
            tidx_train=splits["train"]["tidx"],
            tidx_val=splits["val"]["tidx"],
            patience=patience,
        )
        trainer_i.fit_calibration(
            splits["val"]["X"], splits["val"]["y"].astype(int),
            tidx_val=splits["val"]["tidx"],
        )
        trainers.append(trainer_i)

    predictor = SeedEnsemble(trainers) if len(trainers) > 1 else trainers[0]

    print("\n--- Evaluation (seed-ensemble + isotonic) ---")
    test_proba = _prob_predict(predictor, splits["test"]["X"], splits["test"]["tidx"])
    test_pred = (test_proba >= 0.5).astype(int)
    y_test = splits["test"]["y"].astype(int)
    test_acc = float((test_pred == y_test).mean()) if len(y_test) else 0.0
    up_total = int((y_test == 1).sum())
    up_correct = int(((test_pred == 1) & (y_test == 1)).sum())
    down_total = int((y_test == 0).sum())
    down_correct = int(((test_pred == 0) & (y_test == 0)).sum())
    print(f"  Test accuracy  : {test_acc:.4f}  ({int((test_pred == y_test).sum())}/{len(y_test)})")
    print(f"  Up   recall    : {up_correct}/{up_total}  "
          f"({up_correct / max(up_total, 1):.2%})")
    print(f"  Down recall    : {down_correct}/{down_total}  "
          f"({down_correct / max(down_total, 1):.2%})")

    # Save primary model (seed 0) as lstm_model.pt; additional seeds next to it.
    trainers[0].save(model_path)
    print(f"\n  Primary model saved: {model_path}")
    for i, t in enumerate(trainers[1:], start=2):
        p = model_path.with_name(f"lstm_model_seed{i}.pt")
        t.save(p)
        print(f"  Seed model saved   : {p}")

    print("\n--- Exporting predictions (seed-averaged) ---")
    save_predictions_csv(predictor, splits, csv_path)
    print(f"  CSV saved:   {csv_path}")

    if not args.no_db_export:
        n = upsert_predictions_db(predictor, splits, DATABASE_PATH)
        print(f"  DB rows:     {n} upserted into predictions table")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Seeds            : {seeds}")
    print(f"  Pos weight       : {pos_weight:.3f}")
    print(f"  Horizon          : {args.horizon}")
    print(f"  Test accuracy    : {test_acc:.4f}")
    print(f"  Random baseline  : 0.5000")
    print(f"  model_name       : {MODEL_NAME}")
    print(f"  model_version    : {MODEL_VERSION}")
    print(f"  model_file       : {model_path}")
    print(f"  predictions      : {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
