"""
NewsDatasetBuilder
==================
Builds a labeled text dataset by:

1. Loading raw news articles from the database (main ``news`` table or the
   supplementary ``news.db`` articles table).
2. Assigning each article a ``prediction_date`` using the 4 PM ET cutoff rule
   with weekend / US-holiday handling.
3. Grouping headlines into one row per (ticker, prediction_date), producing
   ``headlines_text`` (pipe-joined string) and ``headlines_json`` (JSON list)
   plus ``num_articles``.
4. Left-joining to the ``labels`` table on (ticker, prediction_date) to attach
   ``label_binary`` and ``label_return``.

Usage
-----
    from src.data_processing.news_dataset_builder import NewsDatasetBuilder

    builder = NewsDatasetBuilder()          # uses default DATABASE_PATH
    df = builder.build()                    # labeled text dataset
    df_unlabeled = builder.build(require_labels=False)

    # Save to CSV
    builder.save(df, "data/processed/text_dataset.csv")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR
from src.data_processing.standardization import DataStandardizer

logger = logging.getLogger(__name__)


class NewsDatasetBuilder:
    """Build a (ticker, prediction_date) → headlines + label dataset."""

    # Column expected in the main ``news`` table
    _MAIN_NEWS_COLS = {
        "ticker":       "ticker",
        "title":        "headline",
        "published_at": "published_at",
    }

    # Column mapping for the supplementary ``articles`` table in news.db
    _ALT_NEWS_COLS = {
        "ticker":           "ticker",
        "headline":         "headline",
        "published_at_utc": "published_at",
    }

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        news_db_path: Optional[str | Path] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        # Optional secondary news.db (collected via alternative pipeline)
        self.news_db_path = Path(news_db_path) if news_db_path else None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def build(self, require_labels: bool = True) -> pd.DataFrame:
        """
        Build the labeled text dataset.

        Parameters
        ----------
        require_labels : bool
            If True (default), only rows with a matching label are returned
            (inner join).  Set to False to include all (ticker, prediction_date)
            groups even when labels are absent (left join).

        Returns
        -------
        pd.DataFrame with columns:
            ticker, prediction_date, num_articles,
            headlines_text, headlines_json,
            label_binary, label_return,    ← NaN when require_labels=False
            close_t, close_t_plus_1        ← NaN when require_labels=False
        """
        raw = self._load_news()
        if raw.empty:
            logger.warning("No news articles found — returning empty dataset.")
            return pd.DataFrame()

        logger.info("Loaded %d raw news articles.", len(raw))

        # Assign prediction_date via cutoff rule
        raw["prediction_date"] = raw["published_at"].apply(
            self._safe_cutoff
        )
        raw = raw.dropna(subset=["prediction_date"])

        # Group by (ticker, prediction_date)
        grouped = self._group_headlines(raw)
        logger.info(
            "Grouped into %d (ticker, prediction_date) rows.", len(grouped)
        )

        # Join to labels
        labels = self._load_labels()
        if labels.empty:
            logger.warning(
                "Labels table is empty — all label columns will be NaN."
            )

        how = "inner" if require_labels else "left"
        merged = grouped.merge(labels, on=["ticker", "prediction_date"], how=how)

        if require_labels:
            logger.info(
                "After inner join with labels: %d labeled rows.", len(merged)
            )
        else:
            labeled = merged["label_binary"].notna().sum()
            logger.info(
                "After left join: %d rows total, %d with labels.", len(merged), labeled
            )

        return merged.reset_index(drop=True)

    def save(
        self,
        df: pd.DataFrame,
        path: Optional[str | Path] = None,
    ) -> Path:
        """
        Save the dataset to a CSV file.

        Default path: data/processed/text_dataset.csv
        """
        out = Path(path) if path else PROCESSED_DATA_DIR / "text_dataset.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved %d rows to %s", len(df), out)
        return out

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_news(self) -> pd.DataFrame:
        """
        Load news articles from available sources.

        Priority:
        1. Supplementary news.db (``articles`` table) if provided and non-empty.
        2. Main database.db ``news`` table.
        Both are merged if both have data.
        """
        frames: list[pd.DataFrame] = []

        # --- Main DB: news table ---
        if self.db_path.exists():
            conn = sqlite3.connect(self.db_path)
            try:
                df = pd.read_sql(
                    "SELECT ticker, title, published_at FROM news "
                    "WHERE ticker IS NOT NULL AND title IS NOT NULL "
                    "  AND published_at IS NOT NULL",
                    conn,
                )
                if not df.empty:
                    df = df.rename(columns={"title": "headline"})
                    frames.append(df[["ticker", "headline", "published_at"]])
                    logger.info("Main DB: %d articles loaded.", len(df))
            except Exception as exc:
                logger.warning("Could not read main news table: %s", exc)
            finally:
                conn.close()

        # --- Supplementary news.db: articles table ---
        if self.news_db_path and self.news_db_path.exists():
            conn = sqlite3.connect(self.news_db_path)
            try:
                df = pd.read_sql(
                    "SELECT ticker, headline, published_at_utc AS published_at "
                    "FROM articles "
                    "WHERE ticker IS NOT NULL AND headline IS NOT NULL "
                    "  AND published_at_utc IS NOT NULL",
                    conn,
                )
                if not df.empty:
                    frames.append(df[["ticker", "headline", "published_at"]])
                    logger.info("news.db: %d articles loaded.", len(df))
            except Exception as exc:
                logger.warning("Could not read articles table: %s", exc)
            finally:
                conn.close()

        if not frames:
            return pd.DataFrame(columns=["ticker", "headline", "published_at"])

        combined = pd.concat(frames, ignore_index=True)

        # Drop duplicates (same ticker + headline)
        combined = combined.drop_duplicates(subset=["ticker", "headline"])
        return combined

    def _load_labels(self) -> pd.DataFrame:
        """
        Load labels from the database.

        Returns a DataFrame with columns:
            ticker, prediction_date, label_binary, label_return,
            close_t, close_t_plus_1
        """
        if not self.db_path.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql(
                "SELECT ticker, date AS prediction_date, "
                "       label_binary, label_return, "
                "       close_t, close_t_plus_1 "
                "FROM labels "
                "WHERE ticker IS NOT NULL AND date IS NOT NULL",
                conn,
            )
        except Exception as exc:
            logger.warning("Could not read labels table: %s", exc)
            df = pd.DataFrame()
        finally:
            conn.close()

        return df

    @staticmethod
    def _safe_cutoff(published_at: str) -> Optional[str]:
        """Apply cutoff rule, returning None on parse errors."""
        try:
            return DataStandardizer.apply_cutoff_rule(published_at)
        except Exception as exc:
            logger.debug("Cutoff parse failed for %r: %s", published_at, exc)
            return None

    @staticmethod
    def _group_headlines(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate articles into one row per (ticker, prediction_date).

        Produces:
            num_articles     – count of articles
            headlines_text   – headlines joined by " | "
            headlines_json   – JSON array of headline strings
        """
        def agg_fn(group: pd.DataFrame) -> pd.Series:
            headlines = group["headline"].dropna().tolist()
            return pd.Series(
                {
                    "num_articles":    len(headlines),
                    "headlines_text":  " | ".join(headlines),
                    "headlines_json":  json.dumps(headlines, ensure_ascii=False),
                }
            )

        grouped = (
            df.groupby(["ticker", "prediction_date"], sort=True)
            .apply(agg_fn)
            .reset_index()
        )
        return grouped