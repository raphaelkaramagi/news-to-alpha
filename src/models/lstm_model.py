"""Two-layer LSTM with ticker embedding for binary stock movement prediction.

Architecture
    Feature input : (batch, seq_len, num_features)
    Ticker ID     : (batch,)                    - integer per ticker
    Ticker embed  : (batch, seq_len, embed_dim) - learned 4-dim vector per ticker,
                                                   broadcast across timesteps
    Concat        : (batch, seq_len, num_features + embed_dim)
      -> LSTM-1 -> Dropout
      -> LSTM-2 -> Dropout
      -> Linear(hidden -> 1)               # logit (no sigmoid in forward)

Loss: BCEWithLogitsLoss (more numerically stable than BCELoss + sigmoid).
Sigmoid is applied only at inference in predict_proba().
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import LSTM_CONFIG

logger = logging.getLogger(__name__)


class StockLSTM(nn.Module):
    """Stacked 2-layer LSTM with optional learned ticker embedding."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
        num_tickers: int = 0,
        ticker_embed_dim: int = 4,
    ):
        super().__init__()
        hs = hidden_sizes or LSTM_CONFIG["lstm_units"]

        self.num_tickers = int(num_tickers)
        self.ticker_embed_dim = int(ticker_embed_dim) if self.num_tickers > 0 else 0
        if self.num_tickers > 0:
            self.ticker_embedding = nn.Embedding(self.num_tickers, self.ticker_embed_dim)
        else:
            self.ticker_embedding = None

        lstm_input = input_size + self.ticker_embed_dim
        self.lstm1 = nn.LSTM(lstm_input, hs[0], batch_first=True)
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
            elif "bias" in name and "classifier" not in name:
                param.data.fill_(0)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)
        if self.ticker_embedding is not None:
            nn.init.normal_(self.ticker_embedding.weight, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor,
                ticker_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Return raw logits (apply sigmoid at inference)."""
        if self.ticker_embedding is not None:
            if ticker_idx is None:
                raise ValueError(
                    "ticker_idx is required when the model was built with num_tickers > 0"
                )
            emb = self.ticker_embedding(ticker_idx)
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, emb], dim=-1)

        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        logits = self.classifier(out[:, -1, :])
        return logits.squeeze(-1)


class LSTMTrainer:
    """Training loop + eval + persistence for a StockLSTM."""

    def __init__(self, model: StockLSTM, config: dict | None = None,
                 ticker_to_idx: dict[str, int] | None = None,
                 scaler_state: dict | None = None,
                 pos_weight: float | None = None):
        self.config = config or LSTM_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
        )
        if pos_weight is not None:
            pw = torch.tensor([float(pos_weight)], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.pos_weight = float(pos_weight) if pos_weight is not None else None
        self.history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "val_acc": [],
        }
        self.ticker_to_idx = ticker_to_idx or {}
        self.scaler_state = scaler_state
        self.calibrator = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              tidx_train: np.ndarray | None = None,
              tidx_val: np.ndarray | None = None,
              patience: int = 10) -> dict:
        train_loader = self._make_loader(X_train, y_train, tidx_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, tidx_val, shuffle=False)

        best_val_acc, best_state, wait = 0.0, None, 0
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

        if best_state is not None:
            self.model.load_state_dict(best_state)
        print(f"\n  Best validation accuracy: {best_val_acc:.4f}")
        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 ticker_idx: np.ndarray | None = None,
                 split_name: str = "Test") -> float:
        loader = self._make_loader(X, y, ticker_idx, shuffle=False)
        acc = self._eval_accuracy(loader)
        y_pred = self.predict(X, ticker_idx)

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

    def predict(self, X: np.ndarray,
                ticker_idx: np.ndarray | None = None) -> np.ndarray:
        probs = self.predict_proba(X, ticker_idx)
        return (probs >= 0.5).astype(np.int32)

    def predict_proba(self, X: np.ndarray,
                      ticker_idx: np.ndarray | None = None,
                      apply_calibration: bool = True) -> np.ndarray:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        tidx_t = None
        if self.model.ticker_embedding is not None:
            if ticker_idx is None:
                raise ValueError(
                    "ticker_idx is required: the model has a ticker embedding."
                )
            tidx_t = torch.LongTensor(ticker_idx).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t, tidx_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        if apply_calibration and self.calibrator is not None:
            probs = np.clip(self.calibrator.predict(probs), 1e-4, 1 - 1e-4)
        return probs

    def fit_calibration(self, X_val: np.ndarray, y_val: np.ndarray,
                        tidx_val: np.ndarray | None = None) -> None:
        """Fit an isotonic regression on val probabilities -> calibrated prob."""
        from sklearn.isotonic import IsotonicRegression
        if len(X_val) == 0:
            self.calibrator = None
            return
        raw = self.predict_proba(X_val, tidx_val, apply_calibration=False)
        if len(np.unique(y_val)) < 2:
            self.calibrator = None
            return
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(raw, y_val.astype(float))
        self.calibrator = cal

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "history": self.history,
            "ticker_to_idx": self.ticker_to_idx,
            "num_tickers": self.model.num_tickers,
            "ticker_embed_dim": self.model.ticker_embed_dim,
            "input_size": self.model.lstm1.input_size - self.model.ticker_embed_dim,
            "scaler_state": self.scaler_state,
            "pos_weight": self.pos_weight,
            "calibrator": self.calibrator,
        }, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path,
             input_size: int | None = None) -> "LSTMTrainer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        in_size = input_size or checkpoint.get("input_size", 17)

        model = StockLSTM(
            input_size=in_size,
            hidden_sizes=config["lstm_units"],
            dropout=config["dropout"],
            num_tickers=checkpoint.get("num_tickers", 0),
            ticker_embed_dim=checkpoint.get("ticker_embed_dim", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(
            model, config,
            ticker_to_idx=checkpoint.get("ticker_to_idx", {}),
            scaler_state=checkpoint.get("scaler_state"),
            pos_weight=checkpoint.get("pos_weight"),
        )
        trainer.history = checkpoint.get("history", {})
        trainer.calibrator = checkpoint.get("calibrator")
        return trainer

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in loader:
            X_batch, y_batch, *tidx_batch = batch
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            tidx_t = tidx_batch[0].to(self.device) if tidx_batch else None

            self.optimizer.zero_grad()
            logits = self.model(X_batch, tidx_t)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y_batch.long()).sum().item()
            total += len(y_batch)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _eval_accuracy(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                X_batch, y_batch, *tidx_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                tidx_t = tidx_batch[0].to(self.device) if tidx_batch else None
                logits = self.model(X_batch, tidx_t)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y_batch.long()).sum().item()
                total += len(y_batch)
        return correct / max(total, 1)

    def _make_loader(self, X: np.ndarray, y: np.ndarray,
                     ticker_idx: np.ndarray | None,
                     shuffle: bool = False) -> DataLoader:
        tensors = [
            torch.FloatTensor(X),
            torch.FloatTensor(y),
        ]
        if self.model.ticker_embedding is not None:
            if ticker_idx is None:
                raise ValueError(
                    "ticker_idx is required: model has a ticker embedding."
                )
            tensors.append(torch.LongTensor(ticker_idx))
        dataset = TensorDataset(*tensors)
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
        )
