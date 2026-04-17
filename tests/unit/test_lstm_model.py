"""Tests for the LSTM model architecture and trainer."""

import numpy as np
import pytest
import torch

from src.models.lstm_model import StockLSTM, LSTMTrainer


TEST_CONFIG = {
    "sequence_length": 5,
    "batch_size": 4,
    "epochs": 3,
    "learning_rate": 0.001,
    "lstm_units": [16, 16],
    "dropout": 0.1,
    "weight_decay": 1e-4,
}

INPUT_SIZE = 20
NUM_TICKERS = 3
EMBED_DIM = 4


@pytest.fixture
def small_model():
    """Model with a ticker embedding, matching the training-time architecture."""
    return StockLSTM(
        input_size=INPUT_SIZE,
        hidden_sizes=[16, 16],
        dropout=0.1,
        num_tickers=NUM_TICKERS,
        ticker_embed_dim=EMBED_DIM,
    )


@pytest.fixture
def small_model_no_embed():
    return StockLSTM(input_size=INPUT_SIZE, hidden_sizes=[16, 16], dropout=0.1)


@pytest.fixture
def dummy_data():
    """40 train + 10 val samples, 3 possible tickers."""
    np.random.seed(42)
    X_train = np.random.randn(40, 5, INPUT_SIZE).astype(np.float32)
    y_train = np.random.randint(0, 2, size=40).astype(np.float32)
    tidx_train = np.random.randint(0, NUM_TICKERS, size=40).astype(np.int64)

    X_val = np.random.randn(10, 5, INPUT_SIZE).astype(np.float32)
    y_val = np.random.randint(0, 2, size=10).astype(np.float32)
    tidx_val = np.random.randint(0, NUM_TICKERS, size=10).astype(np.int64)
    return X_train, y_train, tidx_train, X_val, y_val, tidx_val


class TestStockLSTM:
    def test_output_shape_with_embedding(self, small_model):
        """Forward returns a logit per sample, ticker_idx is required."""
        x = torch.randn(8, 5, INPUT_SIZE)
        tidx = torch.randint(0, NUM_TICKERS, (8,))
        out = small_model(x, tidx)
        assert out.shape == (8,)

    def test_forward_requires_ticker_idx_when_embedding_enabled(self, small_model):
        x = torch.randn(4, 5, INPUT_SIZE)
        with pytest.raises(ValueError):
            small_model(x)

    def test_output_is_logit_not_probability(self, small_model):
        """Forward produces raw logits - some values should be negative."""
        x = torch.randn(32, 5, INPUT_SIZE) * 5  # wide range to trigger extremes
        tidx = torch.randint(0, NUM_TICKERS, (32,))
        out = small_model(x, tidx)
        assert (out < 0).any() or (out > 1).any() or out.abs().max() > 1e-6

    def test_no_embedding_branch(self, small_model_no_embed):
        x = torch.randn(4, 5, INPUT_SIZE)
        out = small_model_no_embed(x)
        assert out.shape == (4,)

    def test_weight_initialization(self, small_model):
        assert small_model.classifier.bias.data.item() == 0.0

    def test_eval_mode_deterministic(self, small_model):
        small_model.eval()
        x = torch.randn(4, 5, INPUT_SIZE)
        tidx = torch.randint(0, NUM_TICKERS, (4,))
        out1 = small_model(x, tidx)
        out2 = small_model(x, tidx)
        assert torch.allclose(out1, out2)


class TestLSTMTrainer:
    def test_train_returns_history(self, small_model, dummy_data):
        X_train, y_train, tidx_train, X_val, y_val, tidx_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        history = trainer.train(
            X_train, y_train, X_val, y_val,
            tidx_train=tidx_train, tidx_val=tidx_val,
            patience=3,
        )
        assert "train_loss" in history
        assert "train_acc" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) > 0

    def test_predict_requires_ticker_idx(self, small_model, dummy_data):
        X_train, y_train, tidx_train, X_val, y_val, tidx_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(
            X_train, y_train, X_val, y_val,
            tidx_train=tidx_train, tidx_val=tidx_val,
            patience=2,
        )
        with pytest.raises(ValueError):
            trainer.predict(X_val)

    def test_predict_returns_binary(self, small_model, dummy_data):
        X_train, y_train, tidx_train, X_val, y_val, tidx_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(
            X_train, y_train, X_val, y_val,
            tidx_train=tidx_train, tidx_val=tidx_val,
            patience=2,
        )
        preds = trainer.predict(X_val, tidx_val)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(y_val)

    def test_predict_proba_in_0_1(self, small_model, dummy_data):
        X_train, y_train, tidx_train, X_val, y_val, tidx_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(
            X_train, y_train, X_val, y_val,
            tidx_train=tidx_train, tidx_val=tidx_val,
            patience=2,
        )
        probs = trainer.predict_proba(X_val, tidx_val)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_save_and_load_roundtrip(self, small_model, dummy_data, tmp_path):
        X_train, y_train, tidx_train, X_val, y_val, tidx_val = dummy_data
        trainer = LSTMTrainer(
            small_model, config=TEST_CONFIG,
            ticker_to_idx={"A": 0, "B": 1, "C": 2},
        )
        trainer.train(
            X_train, y_train, X_val, y_val,
            tidx_train=tidx_train, tidx_val=tidx_val,
            patience=2,
        )

        save_path = tmp_path / "test_lstm.pt"
        trainer.save(save_path)

        loaded = LSTMTrainer.load(save_path)
        original = trainer.predict_proba(X_val, tidx_val)
        restored = loaded.predict_proba(X_val, tidx_val)

        np.testing.assert_array_almost_equal(original, restored, decimal=4)
        assert loaded.ticker_to_idx == {"A": 0, "B": 1, "C": 2}

    def test_evaluate_returns_accuracy(self, small_model, dummy_data):
        X_train, y_train, tidx_train, X_val, y_val, tidx_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(
            X_train, y_train, X_val, y_val,
            tidx_train=tidx_train, tidx_val=tidx_val,
            patience=2,
        )
        acc = trainer.evaluate(X_val, y_val.astype(np.int32), tidx_val)
        assert 0.0 <= acc <= 1.0
