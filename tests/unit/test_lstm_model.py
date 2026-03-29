"""Tests for the LSTM model architecture and trainer."""

import numpy as np
import pytest
import torch

from src.models.lstm_model import StockLSTM, LSTMTrainer


# Small config for fast test runs
TEST_CONFIG = {
    "sequence_length": 5,
    "batch_size": 4,
    "epochs": 3,
    "learning_rate": 0.001,
    "lstm_units": [16, 16],
    "dropout": 0.1,
}


@pytest.fixture
def small_model():
    return StockLSTM(input_size=17, hidden_sizes=[16, 16], dropout=0.1)


@pytest.fixture
def dummy_data():
    """Create small random dataset: 40 train + 10 val samples."""
    np.random.seed(42)
    X_train = np.random.randn(40, 5, 17).astype(np.float32)
    y_train = np.random.randint(0, 2, size=40).astype(np.float32)
    X_val = np.random.randn(10, 5, 17).astype(np.float32)
    y_val = np.random.randint(0, 2, size=10).astype(np.float32)
    return X_train, y_train, X_val, y_val


class TestStockLSTM:
    def test_output_shape(self, small_model):
        """Forward pass should produce one probability per sample."""
        x = torch.randn(8, 5, 17)
        out = small_model(x)
        assert out.shape == (8,)

    def test_output_range(self, small_model):
        """Sigmoid output should be in [0, 1]."""
        x = torch.randn(8, 5, 17)
        out = small_model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_weight_initialization(self, small_model):
        """Classifier bias should be initialized to 0."""
        assert small_model.classifier.bias.data.item() == 0.0

    def test_different_input_sizes(self):
        """Model should accept different feature counts."""
        model = StockLSTM(input_size=10, hidden_sizes=[8, 8])
        x = torch.randn(4, 5, 10)
        out = model(x)
        assert out.shape == (4,)

    def test_single_sample(self, small_model):
        """Should handle batch size of 1."""
        x = torch.randn(1, 5, 17)
        out = small_model(x)
        assert out.shape == (1,)

    def test_eval_mode_deterministic(self, small_model):
        """In eval mode, same input should give same output (dropout off)."""
        small_model.eval()
        x = torch.randn(4, 5, 17)
        out1 = small_model(x)
        out2 = small_model(x)
        assert torch.allclose(out1, out2)


class TestLSTMTrainer:
    def test_train_returns_history(self, small_model, dummy_data):
        """Training should return a history dict with loss and accuracy."""
        X_train, y_train, X_val, y_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        history = trainer.train(X_train, y_train, X_val, y_val, patience=3)

        assert "train_loss" in history
        assert "train_acc" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) > 0

    def test_predict_returns_binary(self, small_model, dummy_data):
        """predict() should return 0/1 labels."""
        X_train, y_train, X_val, y_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(X_train, y_train, X_val, y_val, patience=2)

        preds = trainer.predict(X_val)
        assert set(preds).issubset({0, 1})
        assert len(preds) == len(y_val)

    def test_predict_proba_returns_floats(self, small_model, dummy_data):
        """predict_proba() should return floats in [0, 1]."""
        X_train, y_train, X_val, y_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(X_train, y_train, X_val, y_val, patience=2)

        probs = trainer.predict_proba(X_val)
        assert probs.dtype == np.float32 or probs.dtype == np.float64
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_save_and_load(self, small_model, dummy_data, tmp_path):
        """Saved model should produce identical predictions after loading."""
        X_train, y_train, X_val, y_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(X_train, y_train, X_val, y_val, patience=2)

        save_path = tmp_path / "test_lstm.pt"
        trainer.save(save_path)

        loaded = LSTMTrainer.load(save_path, input_size=17)
        original_preds = trainer.predict_proba(X_val)
        loaded_preds = loaded.predict_proba(X_val)

        np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)

    def test_early_stopping(self, small_model, dummy_data):
        """Training with patience=1 should stop early on random data."""
        X_train, y_train, X_val, y_val = dummy_data
        config = {**TEST_CONFIG, "epochs": 100}
        trainer = LSTMTrainer(small_model, config=config)
        history = trainer.train(X_train, y_train, X_val, y_val, patience=1)

        assert len(history["train_loss"]) < 100

    def test_evaluate_returns_accuracy(self, small_model, dummy_data):
        """evaluate() should return a float accuracy in [0, 1]."""
        X_train, y_train, X_val, y_val = dummy_data
        trainer = LSTMTrainer(small_model, config=TEST_CONFIG)
        trainer.train(X_train, y_train, X_val, y_val, patience=2)

        acc = trainer.evaluate(X_val, y_val.astype(np.int32))
        assert 0.0 <= acc <= 1.0
