#!/usr/bin/env python3
"""Inference-only scoring: load saved model artifacts and write prediction CSVs.

Run after collect_prices / collect_news / generate_labels when you want to
advance prediction dates WITHOUT retraining. Reads existing .pt / .joblib
files and writes the same CSVs that the train_* scripts produce.

Outputs (same paths as training)
---------------------------------
  data/processed/price_predictions.csv
  data/processed/news_tfidf_predictions.csv
  data/processed/news_embeddings_predictions.csv

Usage
-----
  python scripts/score_models.py                  # all three models
  python scripts/score_models.py --skip-lstm
  python scripts/score_models.py --dry-run        # report what would run, exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (  # noqa: E402
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    DATABASE_PATH,
    TICKERS,
)
from src.utils.pipeline_config import load_or_default  # noqa: E402


# ---------------------------------------------------------------------------
# LSTM inference
# ---------------------------------------------------------------------------

def _score_lstm(horizon: int, dry_run: bool) -> bool:
    model_path = MODELS_DIR / "lstm_model.pt"

    if not model_path.exists():
        print(f"[score_lstm] SKIP – model not found: {model_path}")
        return False

    if dry_run:
        print(f"[score_lstm] DRY-RUN – would append live rows, horizon={horizon}")
        return True

    from src.ml.lstm_live_export import append_live_lstm_predictions  # noqa: E402

    cfg = load_or_default()
    tickers = cfg.get("tickers") or list(TICKERS)
    n = append_live_lstm_predictions(tickers, horizon=horizon, model_path=model_path)
    print(f"[score_lstm] Appended {n} live rows to price_predictions.csv")
    return n > 0


# ---------------------------------------------------------------------------
# TF-IDF NLP inference
# ---------------------------------------------------------------------------

def _score_tfidf(horizon: int, dry_run: bool) -> bool:
    model_path = MODELS_DIR / "nlp_baseline.joblib"
    out_csv = PROCESSED_DATA_DIR / "news_tfidf_predictions.csv"

    if not model_path.exists():
        print(f"[score_tfidf] SKIP – model not found: {model_path}")
        return False

    print(f"[score_tfidf] Loading model from {model_path}")
    if dry_run:
        print("[score_tfidf] DRY-RUN – would score all news-bearing days")
        return True

    import joblib
    import pandas as pd
    from src.models.news_pipeline import build_dataset  # noqa: E402
    from scripts.train_nlp import (  # noqa: E402
        export_predictions,
        _text_with_snippet,
        _SIDE_FEATURES,
    )
    from scipy import sparse as sp

    model = joblib.load(model_path)
    df = build_dataset(DATABASE_PATH, drop_rows_without_news=True, horizon=horizon)
    if df.empty:
        print("[score_tfidf] No news-bearing rows – skipping.")
        return False

    text = _text_with_snippet(df)
    side = df[[f for f in _SIDE_FEATURES if f in df.columns]].fillna(0.0).values
    tfidf_X = model.vectorizer_.transform(text)
    pub_X = model.publisher_enc_.transform(df)
    import numpy as np
    X = sp.hstack([tfidf_X, pub_X, sp.csr_matrix(side)])
    proba = model.calibrated_.predict_proba(X)[:, 1]

    out_df = pd.DataFrame({
        "ticker": df["ticker"].values,
        "prediction_date": df["label_date"].values,
        "split": df.get("split", pd.Series(["infer"] * len(df))).values,
        "news_pred_proba": proba,
        "news_pred_binary": (proba >= 0.5).astype(int),
        "news_confidence": np.abs(proba - 0.5) * 2.0,
        "actual_binary": df["label_binary"].values,
        "top_headlines": df["headlines_text"].values if "headlines_text" in df.columns else "",
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[score_tfidf] Saved {len(out_df)} rows → {out_csv}")
    return True


# ---------------------------------------------------------------------------
# Embeddings inference
# ---------------------------------------------------------------------------

def _score_embeddings(horizon: int, dry_run: bool) -> bool:
    model_path = MODELS_DIR / "news_embeddings.joblib"
    out_csv = PROCESSED_DATA_DIR / "news_embeddings_predictions.csv"

    if not model_path.exists():
        print(f"[score_embeddings] SKIP – model not found: {model_path}")
        return False

    print(f"[score_embeddings] Loading model from {model_path}")
    if dry_run:
        print("[score_embeddings] DRY-RUN – would score all news-bearing days")
        return True

    import joblib
    import numpy as np
    import pandas as pd
    from src.models.news_pipeline import build_dataset  # noqa: E402

    model = joblib.load(model_path)
    df = build_dataset(DATABASE_PATH, drop_rows_without_news=True, horizon=horizon)
    if df.empty:
        print("[score_embeddings] No news-bearing rows – skipping.")
        return False

    # Embed using the stored encoder
    headlines = df["headlines_text"].fillna("").astype(str).tolist()
    embeddings = model.encoder_.encode(headlines, show_progress_bar=False, batch_size=64)

    # Publisher features
    from src.features.publisher_features import PublisherOneHot  # noqa: E402
    pub_X = model.publisher_enc_.transform(df) if hasattr(model, "publisher_enc_") else np.zeros((len(df), 1))

    # Side features
    side_cols = ["n_headlines", "avg_finnhub_sentiment", "avg_relevance"]
    side = df[[c for c in side_cols if c in df.columns]].fillna(0.0).values
    import scipy.sparse as sp
    X = np.hstack([embeddings, pub_X.toarray() if sp.issparse(pub_X) else pub_X, side])

    proba = model.calibrated_.predict_proba(X)[:, 1]

    out_df = pd.DataFrame({
        "ticker": df["ticker"].values,
        "prediction_date": df["label_date"].values,
        "split": df.get("split", pd.Series(["infer"] * len(df))).values,
        "news_embeddings_pred_proba": proba,
        "news_embeddings_pred_binary": (proba >= 0.5).astype(int),
        "news_embeddings_confidence": np.abs(proba - 0.5) * 2.0,
        "actual_binary": df["label_binary"].values,
        "top_headlines": df["headlines_text"].values if "headlines_text" in df.columns else "",
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[score_embeddings] Saved {len(out_df)} rows → {out_csv}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only model scoring")
    parser.add_argument("--horizon", type=int, default=None, choices=[1, 3],
                        help="Prediction horizon (default: from pipeline_config.json)")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would run without executing.")
    args = parser.parse_args()

    cfg = load_or_default()
    horizon = args.horizon or int(cfg.get("horizon", 1))
    print(f"[score_models] horizon={horizon}  dry_run={args.dry_run}")

    if not args.skip_lstm:
        _score_lstm(horizon, args.dry_run)
    if not args.skip_tfidf:
        _score_tfidf(horizon, args.dry_run)
    if not args.skip_embeddings:
        _score_embeddings(horizon, args.dry_run)

    print("[score_models] Done.")


if __name__ == "__main__":
    main()
