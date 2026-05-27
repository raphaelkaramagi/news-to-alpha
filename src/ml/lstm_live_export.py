"""Append forward-looking LSTM predictions for dates after the last label."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.models.lstm_model import LSTMTrainer

logger = logging.getLogger(__name__)

MODEL_NAME = "lstm_price"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
CSV_NAME = "price_predictions.csv"


def _apply_scaler(X: np.ndarray, scaler_state: dict | None) -> np.ndarray:
    if not scaler_state:
        return X
    from scripts.train_lstm import apply_level_scaler  # noqa: E402
    mean = np.array(scaler_state["mean"], dtype=np.float32)
    std = np.array(scaler_state["std"], dtype=np.float32)
    return apply_level_scaler(X, mean, std)


def _load_predictor(model_path: Path):
    """Load primary + optional seed ensemble from disk."""
    from scripts.train_lstm import SeedEnsemble  # noqa: E402

    trainer = LSTMTrainer.load(model_path)
    seed_files = sorted(model_path.parent.glob("lstm_model_seed*.pt"))
    if seed_files:
        extra = [LSTMTrainer.load(p) for p in seed_files]
        return SeedEnsemble([trainer] + extra)
    return trainer


def append_live_lstm_predictions(
    tickers: list[str],
    horizon: int = 1,
    model_path: Path | None = None,
    csv_path: Path | None = None,
    predictor=None,
    ticker_to_idx: dict[str, int] | None = None,
) -> int:
    """Score unlabeled recent dates and merge into price_predictions.csv.

    Returns number of new rows appended.
    """
    from src.features.sequence_generator import SequenceGenerator

    model_path = model_path or (Path(__file__).resolve().parent.parent.parent / "data" / "models" / "lstm_model.pt")
    csv_path = csv_path or (PROCESSED_DATA_DIR / CSV_NAME)

    if not model_path.exists():
        logger.warning("No LSTM model at %s — skipping live scores", model_path)
        return 0

    if predictor is None:
        predictor = _load_predictor(model_path)
        ticker_to_idx = predictor.trainers[0].ticker_to_idx if hasattr(predictor, "trainers") else predictor.ticker_to_idx
    elif ticker_to_idx is None:
        ticker_to_idx = getattr(predictor, "ticker_to_idx", {}) or {}

    scaler_state = None
    if hasattr(predictor, "trainers"):
        scaler_state = predictor.trainers[0].scaler_state
    else:
        scaler_state = getattr(predictor, "scaler_state", None)

    gen = SequenceGenerator(horizon=horizon)
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        X_live, dates = gen.generate_live(ticker)
        if len(X_live) == 0:
            continue
        X_live = _apply_scaler(X_live, scaler_state)
        tidx = np.full(len(X_live), ticker_to_idx.get(ticker, 0), dtype=np.int64)
        proba = predictor.predict_proba(X_live, tidx)
        binary = (proba >= 0.5).astype(int)
        confidence = np.abs(proba - 0.5) * 2.0

        frames.append(pd.DataFrame({
            "ticker": ticker,
            "prediction_date": dates,
            "split": "live",
            "financial_pred_proba": proba,
            "financial_pred_binary": binary,
            "financial_confidence": confidence,
            "actual_binary": np.nan,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }))
        logger.info("%s: %d live prediction rows (%s .. %s)",
                    ticker, len(dates), dates[0], dates[-1])

    if not frames:
        return 0

    new_rows = pd.concat(frames, ignore_index=True)

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing["prediction_date"] = existing["prediction_date"].astype(str)
        new_rows["prediction_date"] = new_rows["prediction_date"].astype(str)
        # Drop overlapping (ticker, date) from existing, then append fresh live rows
        keys = set(zip(new_rows["ticker"], new_rows["prediction_date"]))
        existing = existing[
            ~existing.apply(
                lambda r: (r["ticker"], r["prediction_date"]) in keys, axis=1
            )
        ]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.sort_values(["ticker", "prediction_date"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    return len(new_rows)
