"""Two-layer LSTM for binary stock movement prediction.

Architecture
    Input (batch, 60, 16)
      → LSTM-1 (50 units) → Dropout
      → LSTM-2 (50 units) → Dropout
      → Linear(50 → 1) → Sigmoid → P(up)

Uses the last hidden state from the second LSTM as input to the
classifier.  Trained with binary cross-entropy and Adam.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import LSTM_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


class StockLSTM(nn.Module):
    """Stacked 2-layer LSTM with dropout for binary classification."""

    def __init__(self, input_size: int = 17,
                 hidden_sizes: list[int] | None = None,
                 dropout: float = 0.3):
        super().__init__()
        hs = hidden_sizes or LSTM_CONFIG["lstm_units"]

        self.lstm1 = nn.LSTM(input_size, hs[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hs[0], hs[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.classifier = nn.Linear(hs[1], 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        # Take only the last timestep's output for classification
        out = self.classifier(out[:, -1, :])
        return torch.sigmoid(out).squeeze(-1)


class LSTMTrainer:
    """Handles training loop, evaluation, and model persistence."""

    def __init__(self, model: StockLSTM, config: dict | None = None):
        self.config = config or LSTM_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"],
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
        )
        self.criterion = nn.BCELoss()
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_acc": [],
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              patience: int = 10) -> dict:
        """
        Full training loop with early stopping on validation accuracy.

        Returns the training history dict.
        """
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        best_val_acc = 0.0
        best_state = None
        wait = 0
        epochs = self.config["epochs"]

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader)
            val_acc = self._eval_accuracy(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            self.scheduler.step(val_acc)

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                marker = " *" if improved else ""
                print(f"  Epoch {epoch + 1:3d}/{epochs}  "
                      f"Loss: {train_loss:.4f}  "
                      f"Train Acc: {train_acc:.4f}  "
                      f"Val Acc: {val_acc:.4f}{marker}")

            if wait >= patience:
                print(f"  Early stopping at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs)")
                break

        # Restore the best-performing weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

        return self.history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 split_name: str = "Test") -> float:
        """Print accuracy and per-class recall for a dataset split."""
        loader = self._make_loader(X, y, shuffle=False)
        acc = self._eval_accuracy(loader)
        y_pred = self.predict(X)

        up_total = int((y == 1).sum())
        up_correct = int(((y_pred == 1) & (y == 1)).sum())
        down_total = int((y == 0).sum())
        down_correct = int(((y_pred == 0) & (y == 0)).sum())

        print(f"\n  {split_name} Results:")
        print(f"    Accuracy:    {acc:.4f}  ({(y_pred == y).sum()}/{len(y)})")
        print(f"    Up   recall: {up_correct}/{up_total} "
              f"({up_correct / max(up_total, 1):.2%})")
        print(f"    Down recall: {down_correct}/{down_total} "
              f"({down_correct / max(down_total, 1):.2%})")
        return acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0/1) for input sequences."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(up) for each input sequence."""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights, config, and training history."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "history": self.history,
        }, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, input_size: int = 17) -> "LSTMTrainer":
        """Load a saved model from disk."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        model = StockLSTM(
            input_size=input_size,
            hidden_sizes=config["lstm_units"],
            dropout=config["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(model, config)
        trainer.history = checkpoint.get("history", {})
        return trainer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
            correct += ((outputs >= 0.5).long() == y_batch.long()).sum().item()
            total += len(y_batch)

        return total_loss / total, correct / total

    def _eval_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                correct += ((outputs >= 0.5).long() == y_batch.long()).sum().item()
                total += len(y_batch)

        return correct / max(total, 1)

    def _make_loader(self, X: np.ndarray, y: np.ndarray,
                     shuffle: bool = False) -> DataLoader:
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y),
        )
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
        )
