"""Append live expected-move predictions for unlabeled sessions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TICKERS
from src.features.technical_indicators import TechnicalIndicators

MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
CSV_NAME = "volatility_predictions.csv"
MODEL_FILE = "volatility_model.joblib"

FEATURE_COLS = [
    "realized_vol_20",
    "atr_rel",
    "bb_width",
    "volume_zscore_20",
    "overnight_gap",
    "vol_ratio_5_20",
    "vix_level",
    "vix_change",
    "market_return_5d",
    "daily_return",
    "bb_position",
    "rsi_norm",
]


def _load_model():
    path = MODELS_DIR / MODEL_FILE
    if not path.exists():
        return None, FEATURE_COLS
    try:
        payload = joblib.load(path)
    except Exception as exc:
        import sklearn
        print(
            f"[volatility_live_export] SKIP – could not load {path.name}: {exc}\n"
            f"  Runtime sklearn {sklearn.__version__} may differ from the training version.\n"
            f"  Fix: retrain (python scripts/train_volatility.py) and republish the bundle,\n"
            f"  or align scikit-learn in requirements-inference.txt with the training host."
        )
        return None, FEATURE_COLS
    return payload.get("model"), payload.get("features", FEATURE_COLS)


def append_live_volatility_from_frame(live_df: pd.DataFrame) -> int:
    """Merge pre-scored live rows into volatility_predictions.csv."""
    if live_df.empty:
        return 0
    csv_path = PROCESSED_DATA_DIR / CSV_NAME
    out = live_df.copy()
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing["prediction_date"] = existing["prediction_date"].astype(str)
        out["prediction_date"] = out["prediction_date"].astype(str)
        keys = set(zip(out["ticker"], out["prediction_date"]))
        existing = existing[
            ~existing.apply(
                lambda r: (r["ticker"], r["prediction_date"]) in keys, axis=1
            )
        ]
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out
    combined.to_csv(csv_path, index=False)
    return len(out)


def append_live_volatility_predictions(
    tickers: list[str] | None = None,
    horizon: int = 1,
) -> int:
    """Score latest price sessions and append to volatility_predictions.csv."""
    model, features = _load_model()
    if model is None:
        return 0

    price_csv = PROCESSED_DATA_DIR / "price_predictions.csv"
    if not price_csv.exists():
        return 0
    price = pd.read_csv(price_csv, usecols=["ticker", "prediction_date", "split"])
    live_pairs = set(
        zip(
            price.loc[price["split"] == "live", "ticker"],
            price.loc[price["split"] == "live", "prediction_date"].astype(str),
        )
    )
    if not live_pairs:
        return 0

    vol_csv = PROCESSED_DATA_DIR / CSV_NAME
    if vol_csv.exists():
        existing = pd.read_csv(vol_csv, usecols=["ticker", "prediction_date"])
        have = set(zip(existing["ticker"], existing["prediction_date"].astype(str)))
        live_pairs -= have
    if not live_pairs:
        return 0

    ti = TechnicalIndicators()
    tickers = tickers or list(TICKERS)
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        df = ti.compute(ticker)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"date": "prediction_date"})
        df["prediction_date"] = df["prediction_date"].dt.strftime("%Y-%m-%d")
        df["ticker"] = ticker
        if "bb_width" not in df.columns and "bb_upper" in df.columns:
            mid = df.get("bb_middle", df["close"])
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid.replace(0, np.nan)
        mask = df.apply(
            lambda r: (r["ticker"], r["prediction_date"]) in live_pairs, axis=1
        )
        sub = df.loc[mask].dropna(subset=features)
        if sub.empty:
            continue
        pred = np.clip(model.predict(sub[features]), 0.05, 20.0)
        frames.append(pd.DataFrame({
            "ticker": sub["ticker"].values,
            "prediction_date": sub["prediction_date"].values,
            "split": "live",
            "expected_move_pct": pred,
            "actual_abs_return_pct": np.nan,
            "model_name": "volatility",
            "model_version": MODEL_VERSION,
        }))

    if not frames:
        return 0
    return append_live_volatility_from_frame(pd.concat(frames, ignore_index=True))
