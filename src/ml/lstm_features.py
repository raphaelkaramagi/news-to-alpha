"""Resolve LSTM feature columns saved in a checkpoint (inference-safe)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.features.sequence_generator import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def feature_columns_for_model(
    model_path: Path,
    predictor=None,
) -> list[str]:
    """Return feature column names aligned with a saved LSTM checkpoint."""
    saved: list[str] | None = None
    input_size: int | None = None

    if predictor is not None:
        saved = getattr(predictor, "feature_columns", None)
        if saved:
            return list(saved)
        if hasattr(predictor, "trainers"):
            saved = getattr(predictor.trainers[0], "feature_columns", None)
            if saved:
                return list(saved)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        saved = checkpoint.get("feature_columns")
        input_size = checkpoint.get("input_size")

    if saved:
        missing = [c for c in saved if c not in FEATURE_COLUMNS]
        if missing:
            raise RuntimeError(
                f"LSTM checkpoint expects columns missing from current code: {missing}. "
                "Retrain with scripts/train_lstm.py or run_pipeline.py."
            )
        return list(saved)

    if input_size is None:
        return list(FEATURE_COLUMNS)

    if input_size == len(FEATURE_COLUMNS):
        return list(FEATURE_COLUMNS)

    if input_size < len(FEATURE_COLUMNS):
        logger.warning(
            "LSTM checkpoint expects %d features; code defines %d — using the first %d "
            "(legacy checkpoint without feature_columns). Retrain recommended.",
            input_size, len(FEATURE_COLUMNS), input_size,
        )
        return list(FEATURE_COLUMNS[:input_size])

    raise RuntimeError(
        f"LSTM checkpoint expects {input_size} features but code only defines "
        f"{len(FEATURE_COLUMNS)}. Retrain: python scripts/train_lstm.py"
    )
