#!/usr/bin/env python3
"""Daily update: collect fresh prices/news, run inference-only scoring, rebuild ensemble.

Does NOT retrain models. Reads saved artifacts from data/models/ and produces
fresh final_ensemble_predictions.csv. Call this from your Mac cron after market
close, then follow with publish_deploy_bundle.py to push to Railway.

Steps
-----
  1. collect_prices.py    --days LOOKBACK
  2. collect_news.py      --days LOOKBACK   (if include_news)
  3. generate_labels.py
  4. score_models.py      (LSTM + TF-IDF + embeddings inference, no train)
  5. build_eval_dataset.py
  6. build_ensemble.py
  7. evaluate_predictions.py

Usage
-----
  python scripts/daily_update.py
  python scripts/daily_update.py --lookback-days 90
  python scripts/daily_update.py --skip-news --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.pipeline_config import load_or_default, save as _save_cfg  # noqa: E402


def _py() -> str:
    return sys.executable


def _run(cmd: list[str], label: str) -> None:
    print(f"\n>> [{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"Step `{label}` exited with code {result.returncode}."
        )


def run(
    lookback_days: int = 60,
    include_news: bool = True,
    skip_lstm: bool = False,
    skip_tfidf: bool = False,
    skip_embeddings: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the daily update pipeline (infer-only)."""
    cfg = load_or_default()
    horizon = int(cfg.get("horizon", 1))
    tickers: list[str] = cfg.get("tickers") or []

    ticker_args = ["--tickers", *tickers] if tickers else []
    horizon_args = ["--horizon", str(horizon)]

    print("=" * 60)
    print("DAILY UPDATE (inference-only)")
    print("=" * 60)
    print(f"  horizon       : {horizon}")
    print(f"  lookback_days : {lookback_days}")
    print(f"  tickers       : {tickers or 'from config'}")
    print(f"  include_news  : {include_news}")
    print(f"  dry_run       : {dry_run}")
    print()

    if dry_run:
        print("[daily_update] DRY-RUN – no commands executed.")
        return

    _run(
        [_py(), "scripts/collect_prices.py", "--days", str(lookback_days), *ticker_args],
        "collect_prices",
    )

    if include_news:
        _run(
            [_py(), "scripts/collect_news.py",
             "--days", str(lookback_days), "--fill-gaps", *ticker_args],
            "collect_news",
        )

    _run(
        [_py(), "scripts/generate_labels.py", *ticker_args],
        "generate_labels",
    )

    score_args = [*horizon_args]
    if skip_lstm:
        score_args.append("--skip-lstm")
    if skip_tfidf:
        score_args.append("--skip-tfidf")
    if skip_embeddings:
        score_args.append("--skip-embeddings")

    _run(
        [_py(), "scripts/score_models.py", *score_args],
        "score_models",
    )

    _run([_py(), "scripts/build_eval_dataset.py"], "build_eval_dataset")
    _run([_py(), "scripts/build_ensemble.py"], "build_ensemble")
    _run([_py(), "scripts/evaluate_predictions.py", *horizon_args], "evaluate_predictions")

    # Mark as inference run (preserves other config fields)
    _save_cfg(cfg, run_type="daily_update")

    print("\n" + "=" * 60)
    print("DAILY UPDATE COMPLETE")
    print("=" * 60)
    print("Next step: python scripts/publish_deploy_bundle.py --target railway")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily inference update")
    parser.add_argument("--lookback-days", type=int, default=60,
                        help="Days of price/news to (re-)collect (default: 60)")
    parser.add_argument("--skip-news", action="store_true",
                        help="Skip news collection (prices only)")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing.")
    args = parser.parse_args()

    run(
        lookback_days=args.lookback_days,
        include_news=not args.skip_news,
        skip_lstm=args.skip_lstm,
        skip_tfidf=args.skip_tfidf,
        skip_embeddings=args.skip_embeddings,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
