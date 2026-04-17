"""Canonical dataset builder for news-based models (TF-IDF + embeddings).

Joins the `labels` and `news` tables under the 4 PM ET cutoff rule from
[docs/session1_contract.md](../../docs/session1_contract.md).

The `labels` table is keyed by the LSTM convention: `date` = t, representing
the direction close(t) -> close(t+1).  We bucket each news item into the
information window it belongs to:

    news_dt is in window  (close(t-1), close(t)]  ->  label_date = t

Which in words:

    News before 4 PM ET on trading day t            -> label_date = t
    News at/after 4 PM ET on trading day t          -> label_date = next trading day after t
    News on a weekend / holiday                     -> label_date = next trading day
                                                      (so Sat/Sun news informs Monday's move)

Both `scripts/train_nlp.py` and `scripts/train_news_embeddings.py` import
`build_dataset` / `chronological_split` from here so the two models are
guaranteed to see identical rows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, date as date_type, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytz

from src.data_processing.standardization import _FIXED_HOLIDAYS

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


def _is_trading_day(d: date_type) -> bool:
    return d.weekday() < 5 and (d.month, d.day) not in _FIXED_HOLIDAYS


def _next_trading_day(d: date_type) -> date_type:
    d = d + timedelta(days=1)
    while not _is_trading_day(d):
        d += timedelta(days=1)
    return d


def _this_or_next_trading_day(d: date_type) -> date_type:
    while not _is_trading_day(d):
        d += timedelta(days=1)
    return d


def _parse_published_at(published_at: str) -> Optional[datetime]:
    """Parse ISO-8601 strings or numeric UNIX timestamps into tz-aware UTC."""
    if not published_at:
        return None
    s = published_at.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    try:
        ts = float(s)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def map_published_to_label_date(
    published_at: str,
    cutoff_hour: int = 16,
) -> Optional[str]:
    """Return the label_date (YYYY-MM-DD) a news article informs, or None.

    See module docstring for the bucketing rule.
    """
    dt_utc = _parse_published_at(published_at)
    if dt_utc is None:
        return None

    dt_et = dt_utc.astimezone(ET)
    d = dt_et.date()

    if _is_trading_day(d):
        if dt_et.hour < cutoff_hour:
            return d.strftime("%Y-%m-%d")
        return _next_trading_day(d).strftime("%Y-%m-%d")

    # Weekend / holiday: bucket into the next trading day.
    return _this_or_next_trading_day(d).strftime("%Y-%m-%d")


def _label_columns(horizon: int) -> tuple[str, str]:
    if horizon == 1:
        return "label_binary", "label_return"
    if horizon == 3:
        return "label_binary_h3", "return_h3"
    raise ValueError(f"Unsupported horizon {horizon}; expected 1 or 3")


def _load_labels(conn: sqlite3.Connection, horizon: int = 1) -> pd.DataFrame:
    label_col, return_col = _label_columns(horizon)
    df = pd.read_sql_query(
        f"SELECT ticker, date AS label_date, "
        f"{label_col} AS label_binary, {return_col} AS label_return FROM labels",
        conn,
    )
    if df.empty:
        return df
    df["label_date"] = pd.to_datetime(df["label_date"]).dt.strftime("%Y-%m-%d")
    return df


EMPTY_NEWS_COLUMNS = [
    "ticker", "label_date", "headlines_text", "top_headlines", "urls",
    "sources", "n_headlines", "content_snippet",
    "avg_finnhub_sentiment", "avg_relevance",
]

_CONTENT_SNIPPET_CHARS = 800


def _load_news_aligned(conn: sqlite3.Connection, cutoff_hour: int = 16) -> pd.DataFrame:
    """Load news, apply cutoff, aggregate one row per (ticker, label_date).

    Emits not just titles but also a relevance-weighted `content_snippet`,
    averaged Finnhub `sentiment_score` / `relevance_score`, and the per-headline
    URL list so the UI can render clickable news cards.
    """
    has_content = {
        row[1] for row in conn.execute("PRAGMA table_info(news)")
    }
    content_col = "content" if "content" in has_content else "NULL"
    sent_col = "sentiment_score" if "sentiment_score" in has_content else "NULL"
    rel_col = "relevance_score" if "relevance_score" in has_content else "NULL"
    url_col = "url" if "url" in has_content else "NULL"

    raw = pd.read_sql_query(
        f"SELECT ticker, title, source, published_at, "
        f"{content_col} AS content, {sent_col} AS sentiment_score, "
        f"{rel_col} AS relevance_score, {url_col} AS url "
        f"FROM news WHERE title IS NOT NULL AND title != ''",
        conn,
    )
    empty = pd.DataFrame(columns=EMPTY_NEWS_COLUMNS)
    if raw.empty:
        logger.warning("news table is empty")
        return empty

    raw["label_date"] = raw["published_at"].apply(
        lambda s: map_published_to_label_date(s, cutoff_hour)
    )
    raw = raw.dropna(subset=["label_date"])
    if raw.empty:
        logger.warning("All news timestamps were unparseable")
        return empty

    raw = raw.sort_values(["ticker", "label_date", "published_at"])

    def _agg(group: pd.DataFrame) -> pd.Series:
        titles = group["title"].dropna().astype(str).tolist()
        sources = group["source"].fillna("unknown").astype(str).tolist()
        urls = group["url"].fillna("").astype(str).tolist()

        sents = pd.to_numeric(group["sentiment_score"], errors="coerce")
        rels = pd.to_numeric(group["relevance_score"], errors="coerce")
        avg_sent = float(sents.mean()) if sents.notna().any() else 0.0
        avg_rel = float(rels.mean()) if rels.notna().any() else 0.0

        content_series = group["content"].fillna("").astype(str)
        weights = rels.fillna(0.0).to_numpy()
        order = (
            np.argsort(-weights) if (weights > 0).any()
            else np.arange(len(content_series))
        )
        snippet_parts: list[str] = []
        total = 0
        for i in order:
            text = content_series.iloc[int(i)].strip()
            if not text:
                continue
            remaining = _CONTENT_SNIPPET_CHARS - total
            if remaining <= 0:
                break
            snippet_parts.append(text[:remaining])
            total += min(len(text), remaining)
        content_snippet = " ".join(snippet_parts)

        return pd.Series({
            "headlines_text": " . ".join(titles),
            "top_headlines": json.dumps(titles),
            "urls": json.dumps(urls),
            "sources": json.dumps(sources),
            "n_headlines": len(titles),
            "content_snippet": content_snippet,
            "avg_finnhub_sentiment": avg_sent,
            "avg_relevance": avg_rel,
        })

    aggregated = (
        raw.groupby(["ticker", "label_date"], as_index=False)
           .apply(_agg, include_groups=False)
           .reset_index(drop=True)
    )
    if isinstance(aggregated.index, pd.MultiIndex):
        aggregated = aggregated.reset_index()
    return aggregated


def build_dataset(
    db_path: str | Path,
    cutoff_hour: int = 16,
    drop_rows_without_news: bool = True,
    horizon: int = 1,
) -> pd.DataFrame:
    """Join labels with cutoff-aligned news to produce the modeling dataset.

    Unlike the previous implementation, rows with zero headlines are dropped
    by default (no `__NO_NEWS__` placeholder that leaks a has-news flag into
    TF-IDF).  Set `drop_rows_without_news=False` to keep them (useful for
    predicting on ALL ticker-days at inference time).

    `horizon` picks either the 1-day or 3-day label column.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        labels = _load_labels(conn, horizon=horizon)
        news = _load_news_aligned(conn, cutoff_hour=cutoff_hour)
    finally:
        conn.close()

    if labels.empty:
        raise RuntimeError(
            "labels table is empty.  Run scripts/generate_labels.py first."
        )

    empty_list_json = json.dumps([])
    if news.empty:
        df = labels.copy()
        df["headlines_text"] = ""
        df["top_headlines"] = empty_list_json
        df["urls"] = empty_list_json
        df["sources"] = empty_list_json
        df["n_headlines"] = 0
        df["content_snippet"] = ""
        df["avg_finnhub_sentiment"] = 0.0
        df["avg_relevance"] = 0.0
    else:
        df = labels.merge(
            news, on=["ticker", "label_date"], how="left"
        )
        df["headlines_text"] = df["headlines_text"].fillna("")
        df["top_headlines"] = df["top_headlines"].fillna(empty_list_json)
        df["urls"] = df["urls"].fillna(empty_list_json)
        df["sources"] = df["sources"].fillna(empty_list_json)
        df["n_headlines"] = df["n_headlines"].fillna(0).astype(int)
        df["content_snippet"] = df["content_snippet"].fillna("")
        df["avg_finnhub_sentiment"] = df["avg_finnhub_sentiment"].fillna(0.0).astype(float)
        df["avg_relevance"] = df["avg_relevance"].fillna(0.0).astype(float)

    df = df.dropna(subset=["label_binary"])
    df["label_binary"] = df["label_binary"].astype(int)

    # Rename to match the ensemble / eval contract
    df = df.rename(columns={"label_date": "prediction_date"})

    if drop_rows_without_news:
        with_news = df["n_headlines"] > 0
        dropped = int((~with_news).sum())
        df = df[with_news].reset_index(drop=True)
        logger.info(
            "Dataset: %d rows with news (dropped %d without news), %d tickers",
            len(df), dropped, df["ticker"].nunique(),
        )
    else:
        logger.info(
            "Dataset: %d rows (%d with news), %d tickers",
            len(df),
            int((df["n_headlines"] > 0).sum()),
            df["ticker"].nunique(),
        )

    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by unique prediction_date in chronological order.

    All tickers on the same date land in the same split so a date's news
    never leaks between train/val/test.
    """
    dates = sorted(df["prediction_date"].unique())
    n = len(dates)
    if n < 3:
        raise ValueError(
            f"Need at least 3 unique prediction_dates to split; got {n}. "
            "Run scripts/generate_labels.py and scripts/collect_news.py for more data."
        )

    train_end = max(1, int(n * train_ratio))
    val_end = train_end + max(1, int(n * val_ratio))
    val_end = min(val_end, n - 1)

    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])

    train = df[df["prediction_date"].isin(train_dates)].copy()
    val = df[df["prediction_date"].isin(val_dates)].copy()
    test = df[df["prediction_date"].isin(test_dates)].copy()

    logger.info(
        "Split — train: %d rows (%d dates) | val: %d rows (%d dates) | test: %d rows (%d dates)",
        len(train), len(train_dates),
        len(val), len(val_dates),
        len(test), len(test_dates),
    )
    return train, val, test
