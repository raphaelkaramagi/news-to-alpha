"""Tests for scripts/evaluate_predictions.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location(
        "evaluate_predictions", _ROOT / "scripts" / "evaluate_predictions.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ev = _load()


@pytest.fixture
def preds_and_labels(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 200
    rows = []
    for i in range(n):
        y = int(rng.integers(0, 2))
        # Strong LSTM signal, medium news, noise in others
        lstm = 0.8 if y == 1 else 0.2
        lstm = float(np.clip(lstm + rng.normal(0, 0.05), 0, 1))
        tfidf = float(rng.random())
        emb = float(rng.random())
        ens = 0.5 * lstm + 0.25 * tfidf + 0.25 * emb
        rows.append(dict(
            ticker="AAPL" if i % 2 == 0 else "MSFT",
            prediction_date=f"2025-02-{(i % 28)+1:02d}",
            split="test",
            financial_pred_proba=lstm,
            news_tfidf_pred_proba=tfidf,
            news_embeddings_pred_proba=emb,
            ensemble_pred_proba=float(np.clip(ens, 0, 1)),
            actual_binary=y,
        ))
    preds = pd.DataFrame(rows)
    preds_path = tmp_path / "final_ensemble_predictions.csv"
    preds.to_csv(preds_path, index=False)

    # Labels for previous_day_direction baseline
    labels_rows = []
    for ticker in ("AAPL", "MSFT"):
        for d in range(1, 30):
            labels_rows.append(dict(
                ticker=ticker,
                label_date=f"2025-02-{d:02d}",
                direction=int(rng.integers(0, 2)),
            ))
    labels = pd.DataFrame(labels_rows)
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    return preds_path, labels_path


class TestEvaluate:
    def test_outputs_expected_models(self, preds_and_labels):
        preds_path, labels_path = preds_and_labels
        overall, by_ticker, _ = ev.evaluate_all(preds_path, labels_path, split="test")
        expected = {
            "lstm_price", "news_tfidf", "news_embeddings",
            "ensemble", "always_up", "previous_day_direction",
        }
        assert expected <= set(overall["model"])

    def test_metric_columns(self, preds_and_labels):
        preds_path, labels_path = preds_and_labels
        overall, by_ticker, _ = ev.evaluate_all(preds_path, labels_path, split="test")
        for col in ["accuracy", "precision_up", "recall_up", "f1_up", "auc", "n"]:
            assert col in overall.columns
            assert col in by_ticker.columns

    def test_lstm_beats_random(self, preds_and_labels):
        preds_path, labels_path = preds_and_labels
        overall, _, _ = ev.evaluate_all(preds_path, labels_path, split="test")
        lstm_acc = overall.loc[overall["model"] == "lstm_price", "accuracy"].iloc[0]
        always_up_acc = overall.loc[overall["model"] == "always_up", "accuracy"].iloc[0]
        assert lstm_acc > always_up_acc

    def test_always_up_predicts_class_one(self, preds_and_labels):
        preds_path, labels_path = preds_and_labels
        df = pd.read_csv(preds_path)
        pred, proba = ev._baseline_always_up(df)
        assert (pred == 1).all()
        assert np.allclose(proba, 0.5)

    def test_n_equals_total_rows(self, preds_and_labels):
        preds_path, labels_path = preds_and_labels
        overall, _, _ = ev.evaluate_all(preds_path, labels_path, split="test")
        df = pd.read_csv(preds_path)
        expected_n = (df["split"] == "test").sum()
        for _, r in overall.iterrows():
            assert r["n"] == expected_n
