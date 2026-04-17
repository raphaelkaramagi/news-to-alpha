#!/usr/bin/env python3
"""Single-command orchestrator for the full ML pipeline.

Runs the canonical sequence, with options:
  1. collect_prices.py   --days LOOKBACK [--tickers ...]
  2. collect_news.py     --days LOOKBACK [--tickers ...]
  3. generate_labels.py  [--tickers ...]
  4. split_dataset.py
  5. train_lstm.py       --horizon H --seeds ... [--min-move-pct ...] [--tickers ...]
  6. train_nlp.py        --horizon H [--min-move-pct ...]
  7. train_news_embeddings.py --horizon H [--use-finbert] [--min-move-pct ...]
  8. build_eval_dataset.py
  9. build_ensemble.py
 10. evaluate_predictions.py --horizon H

Use as a CLI (`python scripts/run_pipeline.py --preset balanced`) or from
Python (`from scripts.run_pipeline import run`) to pass the same config that
the Flask landing page constructs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import TICKERS  # noqa: E402


@dataclasses.dataclass
class PipelineConfig:
    tickers: list[str]
    lookback_days: int = 365
    horizon: int = 1
    min_move_pct: float = 0.0
    seeds: list[int] = dataclasses.field(default_factory=lambda: [42])
    use_finbert: bool = False
    skip_collect: bool = False
    skip_news: bool = False
    skip_labels: bool = False
    skip_split: bool = False
    skip_lstm: bool = False
    skip_nlp: bool = False
    skip_emb: bool = False
    skip_ensemble: bool = False
    skip_evaluate: bool = False
    temperature_scale: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


PRESETS: dict[str, dict] = {
    "quick": {
        "tickers": ["AAPL", "TSLA"],
        "lookback_days": 365,
        "horizon": 1,
        "min_move_pct": 0.0,
        "seeds": [42],
        "use_finbert": False,
    },
    "balanced": {
        "tickers": ["AAPL", "NVDA", "TSLA", "META", "JPM"],
        "lookback_days": 730,
        "horizon": 3,
        "min_move_pct": 0.5,
        "seeds": [42, 1337, 2024],
        "use_finbert": False,
    },
    "advanced": {
        "tickers": TICKERS,
        "lookback_days": 730,
        "horizon": 3,
        "min_move_pct": 0.3,
        "seeds": [42, 1337, 2024],
        "use_finbert": True,
    },
}


def _run(cmd: list[str], label: str) -> None:
    print(f"\n>> [{label}] " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Step `{label}` exited with code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}"
        )


def _py() -> str:
    return sys.executable


def _join_tickers(tickers: Iterable[str]) -> list[str]:
    out = []
    for t in tickers:
        out.append(t)
    return out


def run(cfg: PipelineConfig) -> None:
    """Run the full pipeline according to `cfg`. Raises on any step failure."""
    horizon_args = ["--horizon", str(cfg.horizon)]
    move_args = (
        ["--min-move-pct", str(cfg.min_move_pct)] if cfg.min_move_pct > 0 else []
    )
    ticker_args = ["--tickers", *_join_tickers(cfg.tickers)]

    if not cfg.skip_collect:
        _run(
            [_py(), "scripts/collect_prices.py",
             "--days", str(cfg.lookback_days), *ticker_args],
            "collect_prices",
        )
    if not cfg.skip_news:
        news_days = min(cfg.lookback_days, 365)
        _run(
            [_py(), "scripts/collect_news.py",
             "--days", str(news_days), *ticker_args],
            "collect_news",
        )
    if not cfg.skip_labels:
        _run([_py(), "scripts/generate_labels.py", *ticker_args], "generate_labels")
    if not cfg.skip_split:
        _run([_py(), "scripts/split_dataset.py"], "split_dataset")

    if not cfg.skip_lstm:
        seed_args = ["--seeds", *[str(s) for s in cfg.seeds]] if cfg.seeds else []
        _run(
            [_py(), "scripts/train_lstm.py", *horizon_args, *move_args,
             *seed_args, *ticker_args],
            "train_lstm",
        )
    if not cfg.skip_nlp:
        _run(
            [_py(), "scripts/train_nlp.py", *horizon_args, *move_args],
            "train_nlp",
        )
    if not cfg.skip_emb:
        emb_args = [*horizon_args, *move_args]
        if cfg.use_finbert:
            emb_args.append("--use-finbert")
        _run(
            [_py(), "scripts/train_news_embeddings.py", *emb_args],
            "train_news_embeddings",
        )
    if not cfg.skip_ensemble:
        _run([_py(), "scripts/build_eval_dataset.py"], "build_eval_dataset")
        ens_args = []
        if not cfg.temperature_scale:
            ens_args.append("--no-temperature-scale")
        _run([_py(), "scripts/build_ensemble.py", *ens_args], "build_ensemble")
    if not cfg.skip_evaluate:
        _run(
            [_py(), "scripts/evaluate_predictions.py", *horizon_args],
            "evaluate_predictions",
        )


def _build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    preset = PRESETS.get(args.preset or "", {}) if args.preset else {}
    tickers = args.tickers or preset.get("tickers") or TICKERS
    seeds = args.seeds or preset.get("seeds") or [42]
    cfg = PipelineConfig(
        tickers=[t.upper() for t in tickers],
        lookback_days=args.lookback_days or preset.get("lookback_days", 365),
        horizon=args.horizon or preset.get("horizon", 1),
        min_move_pct=(
            args.min_move_pct if args.min_move_pct is not None
            else preset.get("min_move_pct", 0.0)
        ),
        seeds=list(seeds),
        use_finbert=args.use_finbert or preset.get("use_finbert", False),
        skip_collect=args.skip_collect,
        skip_news=args.skip_news,
        skip_labels=args.skip_labels,
        skip_split=args.skip_split,
        skip_lstm=args.skip_lstm,
        skip_nlp=args.skip_nlp,
        skip_emb=args.skip_emb,
        skip_ensemble=args.skip_ensemble,
        skip_evaluate=args.skip_evaluate,
        temperature_scale=not args.no_temperature_scale,
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full news-to-alpha pipeline")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None,
                        help="Start from a preset and override specific fields.")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None, choices=[1, 3])
    parser.add_argument("--min-move-pct", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--use-finbert", action="store_true")
    parser.add_argument("--no-temperature-scale", action="store_true")

    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-nlp", action="store_true")
    parser.add_argument("--skip-emb", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved config and exit.")

    args = parser.parse_args()
    cfg = _build_config_from_args(args)

    print("=" * 70)
    print("PIPELINE CONFIG")
    print("=" * 70)
    print(json.dumps(cfg.to_dict(), indent=2))
    if args.dry_run:
        return
    run(cfg)


if __name__ == "__main__":
    main()
