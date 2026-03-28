"""Logistic-regression baseline on TF-IDF news features.

A simple baseline to check whether news headlines carry predictive signal
for next-day stock movement.  If this beats random (50 %), there's signal
worth extracting with heavier models (FinBERT, embeddings) later.
"""

import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.config import DATABASE_PATH, MODELS_DIR, NLP_CONFIG
from src.features.text_features import TextFeatureExtractor

logger = logging.getLogger(__name__)


class NLPBaseline:
    """Logistic regression on TF-IDF headline features."""

    def __init__(self, db_path: str | Path = DATABASE_PATH,
                 max_features: int | None = None):
        self.db_path = Path(db_path)
        self.max_features = max_features or NLP_CONFIG["max_features"]
        self.extractor = TextFeatureExtractor(
            db_path=self.db_path,
            max_features=self.max_features,
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    def train(self, X_train, y_train) -> None:
        """Fit logistic regression on TF-IDF training features."""
        self.classifier.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.classifier.predict(X_train))
        logger.info("Train accuracy: %.4f", train_acc)

    def evaluate(self, X, y, split_name: str = "Test") -> float:
        """Print classification metrics for a dataset split."""
        y_pred = self.classifier.predict(X)
        acc = accuracy_score(y, y_pred)

        print(f"\n  {split_name} Results:")
        print(f"    Accuracy: {acc:.4f} ({(y_pred == y).sum()}/{len(y)})")
        print(classification_report(
            y, y_pred,
            target_names=["Down", "Up"],
            zero_division=0,
        ))
        return acc

    def predict(self, X) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Return P(up) for each sample."""
        return self.classifier.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the fitted TF-IDF vectorizer and classifier together."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "vectorizer": self.extractor.vectorizer,
            "classifier": self.classifier,
            "max_features": self.max_features,
        }, path)
        logger.info("NLP baseline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path,
             db_path: str | Path = DATABASE_PATH) -> "NLPBaseline":
        """Restore a saved model (vectorizer + classifier)."""
        data = joblib.load(path)
        model = cls(db_path=db_path, max_features=data["max_features"])
        model.extractor.vectorizer = data["vectorizer"]
        model.extractor._is_fitted = True
        model.classifier = data["classifier"]
        return model
