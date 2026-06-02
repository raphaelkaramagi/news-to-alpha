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
    encoder_model: str | None = None  # None = default MiniLM; "finbert" = ProsusAI/finbert
    conditional_ensemble: bool = False  # fit separate HGB for has_news vs no_news rows
    skip_collect: bool = False
    skip_news: bool = False
    skip_labels: bool = False
    skip_split: bool = False
    skip_lstm: bool = False
    skip_nlp: bool = False
    skip_emb: bool = False
    skip_vol: bool = False
    skip_ensemble: bool = False
    skip_evaluate: bool = False
    temperature_scale: bool = True
    lstm_epochs: int | None = None

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
    "max": {
        "tickers": TICKERS,
        "lookback_days": 1095,
        "horizon": 1,
        "min_move_pct": 0.0,
        "seeds": [42, 1337, 2024],
        "use_finbert": True,
        "lstm_epochs": 120,
    },
    # max_v2 activates all accuracy-improvement changes from the roadmap:
    #   - min_move_pct=0.3: filter near-flat training days to reduce label noise
    #   - encoder_model=finbert: use ProsusAI/finbert (768-d) as primary news encoder
    #   - conditional_ensemble: separate HGB for has_news vs no_news rows
    #   - VIX regime features (collected automatically via collect_prices.py)
    #   - LSTM balanced-accuracy early stopping (always on after lstm_model.py update)
    # The "max" preset is preserved unchanged for production deploy continuity.
    "max_v2": {
        "tickers": TICKERS,
        "lookback_days": 1095,
        "horizon": 1,
        "min_move_pct": 0.3,
        "seeds": [42, 1337, 2024],
        "use_finbert": True,
        "encoder_model": "finbert",
        "conditional_ensemble": True,
        "lstm_epochs": 150,
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


def _append_live_news_scores(cfg: PipelineConfig) -> None:
    """Score live headline days for TF-IDF and embeddings after full retrain."""
    from src.ml.news_live_export import (  # noqa: E402
        append_live_embedding_predictions,
        append_live_tfidf_predictions,
    )

    n_tfidf = n_emb = 0
    if not cfg.skip_nlp:
        n_tfidf = append_live_tfidf_predictions()
        print(f"\n>> [live_news_tfidf] Appended {n_tfidf} live rows")
    if not cfg.skip_emb:
        n_emb = append_live_embedding_predictions()
        print(f"\n>> [live_news_embeddings] Appended {n_emb} live rows")
    if n_tfidf == 0 and n_emb == 0:
        print("\n>> [live_news] No live news rows appended (no headlines or no price live rows)")


def run(cfg: PipelineConfig) -> None:
    """Run the full pipeline according to `cfg`. Raises on any step failure."""
    from src.utils.pipeline_config import save as _save_cfg  # noqa: E402
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
        _run(
            [_py(), "scripts/collect_news.py",
             "--days", str(cfg.lookback_days), "--fill-gaps", *ticker_args],
            "collect_news",
        )
    if not cfg.skip_labels:
        _run([_py(), "scripts/generate_labels.py", *ticker_args], "generate_labels")
    if not cfg.skip_split:
        _run([_py(), "scripts/split_dataset.py"], "split_dataset")

    if not cfg.skip_lstm:
        seed_args = ["--seeds", *[str(s) for s in cfg.seeds]] if cfg.seeds else []
        lstm_args = [_py(), "scripts/train_lstm.py", *horizon_args, *move_args,
                     *seed_args, *ticker_args]
        if cfg.lstm_epochs is not None:
            lstm_args.extend(["--epochs", str(cfg.lstm_epochs)])
        _run(lstm_args, "train_lstm")
    if not cfg.skip_nlp:
        _run(
            [_py(), "scripts/train_nlp.py", *horizon_args, *move_args],
            "train_nlp",
        )
    if not cfg.skip_emb:
        emb_args = [*horizon_args, *move_args]
        if cfg.encoder_model:
            emb_args.extend(["--encoder-model", cfg.encoder_model])
        # Sentiment aggregates duplicate signal when FinBERT is already the encoder.
        if cfg.use_finbert and cfg.encoder_model not in ("finbert", "FinBERT"):
            emb_args.append("--use-finbert")
        _run(
            [_py(), "scripts/train_news_embeddings.py", *emb_args],
            "train_news_embeddings",
        )

    # News trainers only score labeled rows; append live rows for the latest session
    # (mirrors LSTM live export in train_lstm.py / score_models.py).
    if not (cfg.skip_nlp and cfg.skip_emb):
        _append_live_news_scores(cfg)

    if not cfg.skip_vol:
        vol_args = [*horizon_args]
        _run([_py(), "scripts/train_volatility.py", *vol_args], "train_volatility")

    if not cfg.skip_ensemble:
        _run([_py(), "scripts/build_eval_dataset.py"], "build_eval_dataset")
        ens_args = []
        if not cfg.temperature_scale:
            ens_args.append("--no-temperature-scale")
        if cfg.conditional_ensemble:
            ens_args.append("--conditional")
        _run([_py(), "scripts/build_ensemble.py", *ens_args], "build_ensemble")
    if not cfg.skip_evaluate:
        _run(
            [_py(), "scripts/evaluate_predictions.py", *horizon_args],
            "evaluate_predictions",
        )

    # Save config after a successful full run so infer-only jobs preserve it.
    _save_cfg(cfg.to_dict(), run_type="full_train")

    from src.config import DATABASE_PATH  # noqa: E402
    from src.utils.pipeline_cleanup import prune_predictions_db  # noqa: E402

    pruned = prune_predictions_db(DATABASE_PATH)
    if pruned:
        print(f"\n[cleanup] Pruned {pruned} stale rows from predictions table.")


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
        encoder_model=(
            args.encoder_model if args.encoder_model is not None
            else preset.get("encoder_model")
        ),
        conditional_ensemble=args.conditional_ensemble or preset.get("conditional_ensemble", False),
        skip_collect=args.skip_collect,
        skip_news=args.skip_news,
        skip_labels=args.skip_labels,
        skip_split=args.skip_split,
        skip_lstm=args.skip_lstm,
        skip_nlp=args.skip_nlp,
        skip_emb=args.skip_emb,
        skip_vol=args.skip_vol,
        skip_ensemble=args.skip_ensemble,
        skip_evaluate=args.skip_evaluate,
        temperature_scale=not args.no_temperature_scale,
        lstm_epochs=(
            args.lstm_epochs if args.lstm_epochs is not None
            else preset.get("lstm_epochs")
        ),
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full news-to-alpha pipeline")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None,
                        help=(
                            "Start from a preset and override specific fields. "
                            "Use 'max_v2' for all accuracy-improvement upgrades "
                            "(FinBERT encoder, conditional ensemble, min_move_pct=0.3)."
                        ))
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None, choices=[1, 3])
    parser.add_argument("--min-move-pct", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--use-finbert", action="store_true")
    parser.add_argument(
        "--encoder-model", default=None,
        help="News embedding encoder: 'finbert' (ProsusAI/finbert 768-d) or 'minilm' (default).",
    )
    parser.add_argument(
        "--conditional-ensemble", action="store_true",
        help="Fit separate HGB meta-models for has_news and no_news rows.",
    )
    parser.add_argument("--lstm-epochs", type=int, default=None,
                        help="Override LSTM training epochs (max preset uses 120).")
    parser.add_argument("--no-temperature-scale", action="store_true")

    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-nlp", action="store_true")
    parser.add_argument("--skip-emb", action="store_true")
    parser.add_argument("--skip-vol", action="store_true")
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
