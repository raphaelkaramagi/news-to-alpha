"""FinBERT-based sentiment features for news headlines.

Produces a small, fixed-width numerical feature vector per ticker-day so it
can be horizontally stacked onto TF-IDF (sparse) and MiniLM embeddings
(dense).

Per-row features (12 numbers):
    mean_pos, mean_neg, mean_neu        - average class probs across headlines
    max_pos, max_neg, max_neu           - most extreme per-class score
    n_headlines                         - how many headlines backed this row
    std_pos, std_neg, std_neu           - within-day agreement
    sent_mean_signed = mean_pos - mean_neg
    sent_max_abs_signed = max_pos - max_neg

All zeros when a row has no headlines (so the vector is still well-defined).

The FinBERT model is lazily loaded and cached in-memory per process. Per-headline
sentiment scores are cached in a small SQLite table so reruns are essentially
free.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

FINBERT_MODEL_ID = "ProsusAI/finbert"

SENTIMENT_FEATURE_NAMES: list[str] = [
    "sent_mean_pos", "sent_mean_neg", "sent_mean_neu",
    "sent_max_pos", "sent_max_neg", "sent_max_neu",
    "sent_std_pos", "sent_std_neg", "sent_std_neu",
    "sent_n_headlines",
    "sent_mean_signed", "sent_max_abs_signed",
]
SENTIMENT_VEC_DIM = len(SENTIMENT_FEATURE_NAMES)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


class FinBertSentiment:
    """Lazy FinBERT wrapper with per-headline caching."""

    def __init__(
        self,
        cache_db: str | Path | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.cache_db = Path(cache_db) if cache_db else None
        self._pipeline = None
        self._device = device
        if self.cache_db:
            self._ensure_cache_table()

    def _ensure_cache_table(self) -> None:
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.cache_db)
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS finbert_cache (
                    text_hash TEXT PRIMARY KEY,
                    pos REAL, neg REAL, neu REAL
                )"""
            )
            conn.commit()
        finally:
            conn.close()

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification, AutoTokenizer, pipeline,
        )
        import torch

        if not os.environ.get("HF_HOME"):
            default_cache = Path.cwd() / "data" / ".hf_cache"
            default_cache.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(default_cache)

        tok = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)

        device = self._device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            top_k=None,
            device=device,
            truncation=True,
            max_length=256,
        )
        logger.info("Loaded FinBERT pipeline (device=%s).", device)
        return self._pipeline

    def _cached_scores(self, texts: list[str]) -> dict[str, tuple[float, float, float]]:
        if not self.cache_db:
            return {}
        hashes = [_hash_text(t) for t in texts]
        if not hashes:
            return {}
        placeholders = ",".join("?" for _ in hashes)
        conn = sqlite3.connect(self.cache_db)
        try:
            rows = conn.execute(
                f"SELECT text_hash, pos, neg, neu FROM finbert_cache "
                f"WHERE text_hash IN ({placeholders})",
                hashes,
            ).fetchall()
        finally:
            conn.close()
        return {h: (p, n, u) for h, p, n, u in rows}

    def _store_scores(self, pairs: list[tuple[str, tuple[float, float, float]]]) -> None:
        if not self.cache_db or not pairs:
            return
        conn = sqlite3.connect(self.cache_db)
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO finbert_cache (text_hash, pos, neg, neu) "
                "VALUES (?, ?, ?, ?)",
                [(h, p, n, u) for h, (p, n, u) in pairs],
            )
            conn.commit()
        finally:
            conn.close()

    def score_headlines(self, texts: Iterable[str]) -> np.ndarray:
        """Return an (N, 3) array of (pos, neg, neu) probabilities per headline."""
        text_list = [(t or "").strip() for t in texts]
        if not text_list:
            return np.zeros((0, 3), dtype=np.float32)

        cached = self._cached_scores(text_list)
        missing_idx = [i for i, t in enumerate(text_list)
                       if _hash_text(t) not in cached]

        if missing_idx:
            pipe = self._load_pipeline()
            missing_texts = [text_list[i] for i in missing_idx]
            new_pairs: list[tuple[str, tuple[float, float, float]]] = []
            batch = self.batch_size
            for start in range(0, len(missing_texts), batch):
                chunk = missing_texts[start:start + batch]
                preds = pipe(chunk)
                for txt, per_label in zip(chunk, preds):
                    scores = {p["label"].lower(): p["score"] for p in per_label}
                    triple = (
                        float(scores.get("positive", 0.0)),
                        float(scores.get("negative", 0.0)),
                        float(scores.get("neutral", 0.0)),
                    )
                    new_pairs.append((_hash_text(txt), triple))
            self._store_scores(new_pairs)
            for (h, triple), idx in zip(new_pairs, missing_idx):
                cached[_hash_text(text_list[idx])] = triple

        out = np.zeros((len(text_list), 3), dtype=np.float32)
        for i, t in enumerate(text_list):
            out[i] = cached[_hash_text(t)]
        return out

    def score_ticker_days(
        self,
        headlines_per_row: list[list[str]],
    ) -> np.ndarray:
        """Aggregate FinBERT scores into a (N_rows, SENTIMENT_VEC_DIM) matrix.

        `headlines_per_row` is a list (one entry per ticker-day) of the
        headline strings for that day.
        """
        n_rows = len(headlines_per_row)
        result = np.zeros((n_rows, SENTIMENT_VEC_DIM), dtype=np.float32)
        if n_rows == 0:
            return result

        flat_texts: list[str] = []
        row_slices: list[tuple[int, int]] = []
        for headlines in headlines_per_row:
            start = len(flat_texts)
            flat_texts.extend(headlines)
            row_slices.append((start, len(flat_texts)))

        if flat_texts:
            all_scores = self.score_headlines(flat_texts)
        else:
            all_scores = np.zeros((0, 3), dtype=np.float32)

        for i, (s, e) in enumerate(row_slices):
            scores = all_scores[s:e]
            n = scores.shape[0]
            if n == 0:
                continue
            mean = scores.mean(axis=0)
            mx = scores.max(axis=0)
            std = scores.std(axis=0) if n > 1 else np.zeros(3, dtype=np.float32)
            result[i, 0:3] = mean
            result[i, 3:6] = mx
            result[i, 6:9] = std
            result[i, 9] = float(n)
            result[i, 10] = float(mean[0] - mean[1])
            result[i, 11] = float(mx[0] - mx[1])
        return result
