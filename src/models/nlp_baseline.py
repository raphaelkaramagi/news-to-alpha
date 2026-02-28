"""TF-IDF + Logistic Regression baseline for news headlines.

Trains on a labeled news dataset (CSV in `data/processed/labeled_news.csv`) and
saves the trained model and vectorizer to `data/models/`. Also writes a
predictions CSV to `data/processed/` and inserts news prediction fields into
the `predictions` DB table (columns: `news_pred_proba`, `news_confidence`,
`news_pred_binary`, `model_version`).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, DATABASE_PATH, NLP_CONFIG


class NLPBaseline:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or datetime.now().strftime("nlp_baseline_%Y%m%dT%H%M%S")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_labeled(self, path: str | Path = PROCESSED_DATA_DIR / "labeled_news.csv") -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Labeled news CSV not found at {p}. Run scripts/build_text_dataset.py first.")

        df = pd.read_csv(p)

        # headlines_text may be JSON list (id/text) or a concatenated string; normalize to plain text
        def extract_text(cell):
            if pd.isna(cell):
                return ""
            if isinstance(cell, str) and cell.strip().startswith("["):
                try:
                    items = json.loads(cell)
                    return " \n ".join([it.get("text","") for it in items])
                except Exception:
                    return cell
            return str(cell)

        df["text"] = df["headlines_text"].apply(extract_text) if "headlines_text" in df.columns else df["text"] if "text" in df.columns else ""

        if "label_binary" not in df.columns:
            raise ValueError("Input CSV must contain `label_binary` column.")

        return df

    def train(self, C: float = 1.0, max_features: int | None = None) -> dict:
        df = self._load_labeled()
        texts = df["text"].fillna("").astype(str).tolist()
        y = df["label_binary"].astype(int).values

        max_f = max_features or NLP_CONFIG.get("max_features")
        vect = TfidfVectorizer(max_features=max_f, stop_words="english")
        X = vect.fit_transform(texts)

        clf = LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=1000)
        clf.fit(X, y)

        # Save artifacts
        model_path = Path(MODELS_DIR) / f"{self.model_name}.joblib"
        vec_path = Path(MODELS_DIR) / f"{self.model_name}_vectorizer.joblib"
        joblib.dump(clf, model_path)
        joblib.dump(vect, vec_path)

        result = {
            "model_path": str(model_path),
            "vectorizer_path": str(vec_path),
            "model_version": self.model_name,
            "n_samples": int(len(y)),
        }

        return result

    def predict_and_save(self, model_version: str | None = None, out_csv: str | Path = PROCESSED_DATA_DIR / "predictions_nlp.csv") -> int:
        model_version = model_version or self.model_name
        model_path = Path(MODELS_DIR) / f"{model_version}.joblib"
        vec_path = Path(MODELS_DIR) / f"{model_version}_vectorizer.joblib"
        if not model_path.exists() or not vec_path.exists():
            raise FileNotFoundError("Model or vectorizer not found. Call `train()` first or provide model_version.")

        clf = joblib.load(model_path)
        vect = joblib.load(vec_path)

        df = self._load_labeled()
        texts = df["text"].fillna("").astype(str).tolist()
        X = vect.transform(texts)

        proba = clf.predict_proba(X)[:, 1]
        pred_binary = (proba > 0.5).astype(int)
        confidence = np.maximum(proba, 1 - proba)

        out = pd.DataFrame({
            "ticker": df["ticker"],
            "date": df["date"],
            "news_pred_proba": proba,
            "news_pred_binary": pred_binary,
            "news_confidence": confidence,
            "model_version": model_version,
        })

        out_path = Path(out_csv)
        out.to_csv(out_path, index=False)

        # Insert into predictions table (news fields only)
        conn = sqlite3.connect(str(DATABASE_PATH))
        cur = conn.cursor()
        inserted = 0
        for _, row in out.iterrows():
            try:
                cur.execute(
                    "INSERT OR REPLACE INTO predictions (ticker, date, news_pred_proba, news_confidence, news_pred_binary, model_version) VALUES (?, ?, ?, ?, ?, ?)",
                    (row["ticker"], row["date"], float(row["news_pred_proba"]), float(row["news_confidence"]), int(row["news_pred_binary"]), model_version),
                )
                inserted += 1
            except Exception:
                # continue on failure for robustness
                continue

        conn.commit()
        conn.close()

        return inserted


if __name__ == "__main__":
    trainer = NLPBaseline()
    info = trainer.train()
    print(f"Trained model: {info['model_path']}")
    n = trainer.predict_and_save()
    print(f"Saved and inserted {n} predictions")
