"""Unit tests for the HistGB meta-model in scripts/build_ensemble.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]


def _load_build_ensemble():
    sys.path.insert(0, str(_ROOT))
    path = _ROOT / "scripts" / "build_ensemble.py"
    spec = importlib.util.spec_from_file_location("build_ensemble", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


be = _load_build_ensemble()


def _make_df(n_val: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(100):
        y = int(rng.integers(0, 2))
        rows.append(dict(
            ticker="AAPL",
            prediction_date=f"2025-01-{i+1:02d}",
            split="train",
            financial_pred_proba=rng.random(),
            news_tfidf_pred_proba=rng.random(),
            news_embeddings_pred_proba=rng.random(),
            has_news=1,
            n_headlines=int(rng.integers(0, 10)),
            spy_return_5d=float(rng.normal(0, 1)),
            all_agree=0,
            actual_binary=y,
            top_headlines="[]",
        ))

    # Val: LSTM probas carry signal, rest is noise
    for i in range(n_val):
        y = int(rng.integers(0, 2))
        lstm = 0.75 if y == 1 else 0.25
        lstm += rng.normal(0, 0.05)
        lstm = float(np.clip(lstm, 0.0, 1.0))
        rows.append(dict(
            ticker="AAPL",
            prediction_date=f"2025-03-{(i % 28)+1:02d}",
            split="val",
            financial_pred_proba=lstm,
            news_tfidf_pred_proba=rng.random(),
            news_embeddings_pred_proba=rng.random(),
            has_news=int(rng.integers(0, 2)),
            n_headlines=int(rng.integers(0, 10)),
            spy_return_5d=float(rng.normal(0, 1)),
            all_agree=0,
            actual_binary=y,
            top_headlines="[]",
        ))
    return pd.DataFrame(rows)


class TestMetaFit:
    def test_meta_recovers_lstm_signal(self):
        df = _make_df(n_val=400, seed=1)
        meta = be.fit_meta_model(df, temperature_scale=True)
        assert meta["model"] is not None
        importances = dict(meta["importances"])
        # The LSTM proba should be the most important feature
        top = meta["importances"][0][0]
        assert top == "financial_pred_proba", (
            f"Expected lstm proba to be most important, got: {meta['importances'][:3]}"
        )
        assert importances["financial_pred_proba"] > 0

    def test_output_columns_and_shape(self):
        df = _make_df(n_val=400, seed=2)
        meta = be.fit_meta_model(df, temperature_scale=False)
        out = be.compute_ensemble(df, meta)

        for col in [
            "ensemble_pred_proba", "ensemble_pred_binary",
            "ensemble_confidence", "model_version",
        ]:
            assert col in out.columns
        assert len(out) == len(df)
        assert out["ensemble_pred_proba"].between(0.0, 1.0).all()
        assert set(out["ensemble_pred_binary"].unique()) <= {0, 1}
        assert out["ensemble_confidence"].between(0.0, 1.0).all()

    def test_uniform_fallback_when_val_too_small(self):
        """With <40 labeled val rows (and no test), fit uses the uniform fallback."""
        rows = []
        for i in range(10):
            rows.append(dict(
                ticker="AAPL",
                prediction_date=f"2025-01-{i+1:02d}",
                split="val",
                financial_pred_proba=0.8,
                news_tfidf_pred_proba=0.9,
                news_embeddings_pred_proba=0.7,
                has_news=1,
                n_headlines=3,
                spy_return_5d=0.0,
                all_agree=1,
                actual_binary=1,
                top_headlines="[]",
            ))
        df = pd.DataFrame(rows)
        meta = be.fit_meta_model(df, temperature_scale=False)
        assert isinstance(meta["model"], be._UniformFallback)
        X = be._feature_matrix(df)
        proba = meta["model"].predict_proba(X)[:, 1]
        expected = X[:, [0, 2, 4]].mean(axis=1)
        np.testing.assert_allclose(proba, expected, rtol=1e-6)

    def test_temperature_scaling_preserves_ranking(self):
        """Temperature scaling should not invert the ranking of two predictions."""
        df = _make_df(n_val=400, seed=5)
        meta = be.fit_meta_model(df, temperature_scale=True)
        out = be.compute_ensemble(df, meta)
        # Monotone: for any two rows, higher raw HistGB+isotonic prob -> higher ensemble prob
        X = be._feature_matrix(df)
        raw = meta["model"].predict_proba(X)[:, 1]
        scaled = out["ensemble_pred_proba"].to_numpy()
        # Rank correlation should be near 1
        from scipy.stats import spearmanr
        rho, _ = spearmanr(raw, scaled)
        assert rho > 0.99
