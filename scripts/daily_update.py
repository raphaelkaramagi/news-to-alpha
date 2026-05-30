#!/usr/bin/env python3
"""Daily update: collect fresh prices/news, run inference-only scoring, rebuild ensemble.

Does NOT retrain models. Reads saved artifacts from data/models/ and produces
fresh final_ensemble_predictions.csv.

Steps
-----
  1. collect_prices.py    (incremental by default)
  2. collect_news.py      (incremental by default, if include_news)
  3. generate_labels.py
  4. score_models.py      (LSTM + TF-IDF + embeddings inference, no train)
  5. build_eval_dataset.py
  6. build_ensemble.py
  7. evaluate_predictions.py

Usage
-----
  python scripts/daily_update.py
  python scripts/daily_update.py --full-lookback          # fixed 60-day window
  python scripts/daily_update.py --lookback-days 90       # max gap catch-up
  python scripts/daily_update.py --skip-news --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH  # noqa: E402
from src.utils.collection_window import compute_collection_window, universe_tickers  # noqa: E402
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


def _print_collection_plan(
    incremental: bool,
    lookback_days: int,
    buffer_days: int,
    min_days: int,
    all_tickers: list[str],
) -> dict | None:
    if not incremental:
        print(f"  collection    : fixed {lookback_days}-day lookback")
        return None
    start, end, info = compute_collection_window(
        all_tickers,
        buffer_days=buffer_days,
        min_days=min_days,
        max_days=lookback_days,
    )
    print(f"  collection    : incremental ({info['mode']})")
    print(f"  window        : {start} → {end}  ({info['lookback_calendar_days']} calendar days)")
    print(f"  price_latest  : {info.get('price_latest') or 'none'}")
    print(f"  news_latest   : {info.get('news_latest') or 'none'}")
    print(f"  gap_days      : {info.get('gap_since_price_latest')}")
    return info


def run(
    lookback_days: int = 60,
    buffer_days: int = 5,
    min_days: int = 7,
    incremental: bool = True,
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
    all_tickers = universe_tickers(tickers)

    ticker_args = ["--tickers", *tickers] if tickers else []
    horizon_args = ["--horizon", str(horizon)]

    print("=" * 60)
    print("DAILY UPDATE (inference-only)")
    print("=" * 60)
    print(f"  horizon       : {horizon}")
    print(f"  max_lookback  : {lookback_days} calendar days")
    print(f"  incremental   : {incremental}")
    print(f"  tickers       : {tickers or 'from config'}")
    print(f"  include_news  : {include_news}")
    print(f"  dry_run       : {dry_run}")

    window_info = _print_collection_plan(
        incremental, lookback_days, buffer_days, min_days, all_tickers,
    )
    print()

    if dry_run:
        print("[daily_update] DRY-RUN – no commands executed.")
        return

    collect_flags = ["--days", str(lookback_days)]
    if incremental and window_info:
        collect_flags = [
            "--since", window_info["start_date"],
            "--days", str(lookback_days),
        ]
    elif not incremental:
        collect_flags = ["--days", str(lookback_days)]

    _run(
        [_py(), "scripts/collect_prices.py", *collect_flags, *ticker_args],
        "collect_prices",
    )

    if include_news:
        _run(
            [_py(), "scripts/collect_news.py",
             *collect_flags, "--fill-gaps", *ticker_args],
            "collect_news",
        )

    _run(
        [_py(), "scripts/generate_labels.py", *ticker_args],
        "generate_labels",
    )

    score_args = [*horizon_args, "--incremental"]
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

    payload = dict(cfg)
    payload["last_daily_update"] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "incremental": incremental,
        "lookback_days": lookback_days,
        "collection_window": window_info,
    }
    _save_cfg(payload, run_type="daily_update")

    print("\n" + "=" * 60)
    print("DAILY UPDATE COMPLETE")
    print("=" * 60)
    print("Check:")
    print("  python scripts/audit_data_coverage.py")
    print("  curl -s http://127.0.0.1:8000/api/data-status | python3 -m json.tool")
    if str(DATABASE_PATH).startswith("/data"):
        print("Data written to Railway /data volume — no publish step needed.")
    else:
        print("Next: python scripts/publish_deploy_bundle.py --target railway --service web")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily inference update")
    parser.add_argument(
        "--lookback-days", type=int, default=60,
        help="Max calendar lookback (default: 60). With --incremental, actual "
             "window may be smaller when data is current.",
    )
    parser.add_argument(
        "--full-lookback", action="store_true",
        help="Disable incremental mode; always fetch --lookback-days.",
    )
    parser.add_argument(
        "--buffer-days", type=int, default=5,
        help="Overlap when incremental (default: 5).",
    )
    parser.add_argument(
        "--min-days", type=int, default=7,
        help="Minimum incremental window (default: 7).",
    )
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
        buffer_days=args.buffer_days,
        min_days=args.min_days,
        incremental=not args.full_lookback,
        include_news=not args.skip_news,
        skip_lstm=args.skip_lstm,
        skip_tfidf=args.skip_tfidf,
        skip_embeddings=args.skip_embeddings,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
