#!/usr/bin/env python3
"""Train expected-move (next-day |return|) regressor from price features.

Target: absolute close-to-close return over the next session, in percent.
Outputs volatility_predictions.csv and volatility_model.joblib.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, TICKERS  # noqa: E402
from src.features.technical_indicators import TechnicalIndicators  # noqa: E402

MODEL_NAME = "volatility"
MODEL_VERSION = datetime.now().strftime("%Y%m%dT%H%M%S")
MODEL_FILE = "volatility_model.joblib"
CSV_NAME = "volatility_predictions.csv"

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

# Free extra features (earnings proximity + sector-relative vol). Gated: only
# kept if they improve test MAE / high-move AUC in this script's eval.
EXTRA_FEATURE_COLS = [
    "days_to_earnings",
    "earnings_window",
    "sector_vol_ratio",
]


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Augment with earnings-proximity and sector-relative realized vol."""
    from src.features.fundamentals_features import add_earnings_proximity, add_sector

    out = add_earnings_proximity(df, date_col="prediction_date")
    out = add_sector(out)
    # Sector-relative vol: this ticker's realized_vol_20 vs same-day sector mean.
    sector_mean = (
        out.groupby(["prediction_date", "sector"])["realized_vol_20"].transform("mean")
    )
    out["sector_vol_ratio"] = out["realized_vol_20"] / sector_mean.replace(0, np.nan)
    out["sector_vol_ratio"] = out["sector_vol_ratio"].fillna(1.0)
    return out


def _load_split_dates() -> dict[str, set[str]]:
    path = PROCESSED_DATA_DIR / "split_info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/split_dataset.py first.")
    with open(path) as f:
        info = json.load(f)
    return {name: set(info["splits"][name]["dates"]) for name in info["splits"]}


def build_volatility_frame(tickers: list[str], horizon: int = 1) -> pd.DataFrame:
    """One row per (ticker, date) with features and realized |return| label."""
    ti = TechnicalIndicators()
    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        df = ti.compute(ticker)
        if df.empty:
            continue
        df = df.reset_index().rename(columns={"date": "prediction_date"})
        df["prediction_date"] = df["prediction_date"].dt.strftime("%Y-%m-%d")
        df["close_fwd"] = df["close"].shift(-horizon)
        df["fwd_return_pct"] = (df["close_fwd"] / df["close"] - 1.0) * 100.0
        df["abs_return_pct"] = df["fwd_return_pct"].abs()
        df["ticker"] = ticker
        df["range_pct"] = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)) * 100.0
        if "bb_width" not in df.columns and "bb_upper" in df.columns and "bb_lower" in df.columns:
            mid = df.get("bb_middle", df["close"])
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid.replace(0, np.nan)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=FEATURE_COLS + ["abs_return_pct"])


def assign_splits(df: pd.DataFrame, split_dates: dict[str, set[str]]) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "train"
    for name in ("val", "test"):
        if name in split_dates:
            df.loc[df["prediction_date"].isin(split_dates[name]), "split"] = name
    # Dates after last labeled session → live (no abs_return_pct yet)
    labeled = df["abs_return_pct"].notna()
    max_labeled = df.loc[labeled, "prediction_date"].max() if labeled.any() else None
    if max_labeled:
        df.loc[~labeled & (df["prediction_date"] > max_labeled), "split"] = "live"
    return df


def train_model(
    train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str],
) -> HistGradientBoostingRegressor:
    reg = HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=400,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    reg.fit(train_df[feature_cols], train_df["abs_return_pct"])
    if not val_df.empty:
        val_pred = reg.predict(val_df[feature_cols])
        print(f"  Val MAE: {mean_absolute_error(val_df['abs_return_pct'], val_pred):.4f}%")
    return reg


def evaluate(model, df: pd.DataFrame, name: str, feature_cols: list[str]) -> dict:
    if df.empty:
        return {}
    pred = np.clip(model.predict(df[feature_cols]), 0.05, 20.0)
    mae = mean_absolute_error(df["abs_return_pct"], pred)
    med = float(df["abs_return_pct"].median())
    y_high = (df["abs_return_pct"] > med).astype(int)
    try:
        auc = float(roc_auc_score(y_high, pred))
    except ValueError:
        auc = float("nan")
    print(f"  {name:5s}  n={len(df):4d}  MAE={mae:.3f}%  high-move AUC={auc:.3f}")
    return {"mae": mae, "high_move_auc": auc, "n": len(df)}


def export_predictions(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    pred = np.clip(model.predict(df[feature_cols]), 0.05, 20.0)
    out = df[["ticker", "prediction_date", "split"]].copy()
    out["expected_move_pct"] = pred
    out["actual_abs_return_pct"] = df["abs_return_pct"].values
    out["model_name"] = MODEL_NAME
    out["model_version"] = MODEL_VERSION
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train volatility / expected-move model")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 3])
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument(
        "--extra-features", action="store_true",
        help="Add earnings-proximity + sector-relative vol (gated; needs collect_fundamentals.py)",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in (args.tickers or TICKERS)]
    split_dates = _load_split_dates()

    print("=" * 70)
    print("VOLATILITY / EXPECTED-MOVE MODEL")
    print("=" * 70)

    df = build_volatility_frame(tickers, horizon=args.horizon)
    print(f"  Built {len(df):,} feature rows across {df['ticker'].nunique()} tickers")

    feature_cols = list(FEATURE_COLS)
    if args.extra_features:
        df = add_extra_features(df)
        feature_cols = feature_cols + EXTRA_FEATURE_COLS
        print(f"  Extra features ON: {EXTRA_FEATURE_COLS}")

    df = assign_splits(df, split_dates)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    print(f"  Train {len(train):,}  Val {len(val):,}  Test {len(test):,}")

    model = train_model(train, val, feature_cols)
    print("\nEvaluation:")
    evaluate(model, train, "train", feature_cols)
    evaluate(model, val, "val", feature_cols)
    evaluate(model, test, "test", feature_cols)

    # Score all rows including live (features known, label may be NaN)
    all_scored = export_predictions(model, df, feature_cols)
    csv_path = PROCESSED_DATA_DIR / CSV_NAME
    labeled = all_scored[all_scored["split"].isin(["train", "val", "test"])]
    combined = labeled.copy()
    combined.to_csv(csv_path, index=False)
    print(f"\n  Saved {len(combined):,} rows to {csv_path}")

    # save joblib before live export — append_live reads features from disk
    model_path = MODELS_DIR / MODEL_FILE
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": feature_cols,
        "horizon": args.horizon,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "high_vol_median_pct": float(train["abs_return_pct"].median()) if not train.empty else 1.0,
    }, model_path)
    print(f"  Saved model to {model_path}")

    from src.ml.volatility_live_export import append_live_volatility_predictions  # noqa: E402
    n_live = append_live_volatility_predictions(tickers=tickers, horizon=args.horizon)
    if n_live:
        print(f"  Appended {n_live} live volatility rows")
    print("=" * 70)


if __name__ == "__main__":
    main()
