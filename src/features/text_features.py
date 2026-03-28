"""Extract TF-IDF features from news headlines for NLP modeling.

For each (ticker, date) pair that has a label, we gather recent headlines
and convert them to a TF-IDF vector.  Pairs with no news coverage are
excluded — the NLP model only predicts when it has text input.

Usage:
    extractor = TextFeatureExtractor()
    texts, labels, metadata = extractor.prepare(tickers)

    X_train = extractor.fit_transform(train_texts)
    X_val   = extractor.transform(val_texts)
"""

import sqlite3
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATABASE_PATH, NLP_CONFIG, TICKERS

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Build TF-IDF features from news headlines aligned with trading labels."""

    def __init__(self, db_path: str | Path = DATABASE_PATH,
                 max_features: int | None = None,
                 lookback_days: int = 3):
        self.db_path = Path(db_path)
        self.max_features = max_features or NLP_CONFIG["max_features"]
        self.lookback_days = lookback_days
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self._is_fitted = False

    def prepare(self, tickers: list[str] | None = None
                ) -> tuple[list[str], np.ndarray, list[tuple[str, str]]]:
        """
        Build text corpus aligned with labels.

        For each (ticker, date) that has both a label and news coverage,
        concatenates headlines from the previous `lookback_days` into one
        string.

        Returns:
            texts:    list of headline strings (one per sample)
            labels:   array of 0/1 labels
            metadata: list of (ticker, date) identifying each sample
        """
        tickers = tickers or TICKERS
        conn = sqlite3.connect(self.db_path)

        ph = ",".join("?" for _ in tickers)
        labels_df = pd.read_sql_query(
            f"SELECT ticker, date, label_binary FROM labels "
            f"WHERE ticker IN ({ph}) ORDER BY date",
            conn, params=tickers,
        )
        news_df = pd.read_sql_query(
            f"SELECT ticker, title, published_at FROM news "
            f"WHERE ticker IN ({ph})",
            conn, params=tickers,
        )
        conn.close()

        if labels_df.empty:
            logger.warning("No labels found — run generate_labels.py first")
            return [], np.array([]), []

        if news_df.empty:
            logger.warning("No news found — run collect_news.py first")
            return [], np.array([]), []

        # Parse the date portion of published_at for date-range matching
        news_df["news_date"] = pd.to_datetime(
            news_df["published_at"].str[:10], errors="coerce"
        )
        news_df = news_df.dropna(subset=["news_date"])
        news_df["news_date"] = news_df["news_date"].dt.strftime("%Y-%m-%d")

        texts: list[str] = []
        valid_labels: list[int] = []
        metadata: list[tuple[str, str]] = []

        for _, row in labels_df.iterrows():
            ticker, date_str, label = row["ticker"], row["date"], row["label_binary"]
            headlines = self._gather_headlines(news_df, ticker, date_str)

            if not headlines:
                continue

            texts.append(" ".join(headlines))
            valid_labels.append(label)
            metadata.append((ticker, date_str))

        skipped = len(labels_df) - len(texts)
        logger.info("Prepared %d samples from %d labels (%d skipped — no news)",
                     len(texts), len(labels_df), skipped)

        return texts, np.array(valid_labels, dtype=np.int32), metadata

    def fit(self, texts: list[str]) -> "TextFeatureExtractor":
        """Fit TF-IDF vocabulary on training texts only."""
        self.vectorizer.fit(texts)
        self._is_fitted = True
        logger.info("TF-IDF fitted: %d features", len(self.vectorizer.vocabulary_))
        return self

    def transform(self, texts: list[str]):
        """Transform texts into a TF-IDF sparse matrix."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        """Fit vocabulary and transform in one step."""
        self._is_fitted = True
        X = self.vectorizer.fit_transform(texts)
        logger.info("TF-IDF fit_transform: %d samples, %d features",
                     X.shape[0], X.shape[1])
        return X

    def _gather_headlines(self, news_df: pd.DataFrame,
                          ticker: str, date_str: str) -> list[str]:
        """Collect non-empty headlines for *ticker* within the lookback window."""
        target = pd.to_datetime(date_str)
        start = (target - pd.Timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

        mask = (
            (news_df["ticker"] == ticker)
            & (news_df["news_date"] >= start)
            & (news_df["news_date"] <= date_str)
        )
        return [h for h in news_df.loc[mask, "title"].tolist() if h and h.strip()]
