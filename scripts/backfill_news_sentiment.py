#!/usr/bin/env python3
"""Populate news.sentiment_score and news.relevance_score.

These columns exist in the schema but the Finnhub collector never wrote them,
so they were always NULL -> the news aggregation produced avg_finnhub_sentiment
= 0 and avg_relevance = 0 for every row. That made two model inputs dead and
collapsed the embedding relevance weighting to a uniform mean.

What this writes
----------------
- sentiment_score : FinBERT signed sentiment in [-1, 1]  (P(pos) - P(neg)),
                    cached per-headline so reruns are nearly free.
- relevance_score : mention-strength heuristic in [0, 1] based on whether the
                    ticker symbol / company name appears in the title vs body.

Usage
-----
  python scripts/backfill_news_sentiment.py            # only NULL rows
  python scripts/backfill_news_sentiment.py --all      # recompute everything
  python scripts/backfill_news_sentiment.py --limit 500 --dry-run
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATABASE_PATH, PROCESSED_DATA_DIR, TICKER_TO_COMPANY  # noqa: E402


def _relevance(ticker: str, title: str, content: str) -> float:
    """Mention-strength relevance in [0, 1] (free, deterministic).

    finnhub doesn't give us a relevance field — this is a cheap proxy so embedding
    pooling weights aren't uniform. title mention > body mention > generic article.
    """
    company = (TICKER_TO_COMPANY.get(ticker, "") or "").lower()
    title_l = (title or "").lower()
    body_l = (content or "").lower()

    if len(ticker) <= 3:
        sym = re.compile(rf"\b{re.escape(ticker.lower())}\b")
    else:
        sym = re.compile(re.escape(ticker.lower()))

    if sym.search(title_l):
        return 1.0
    if company and company in title_l:
        return 0.85
    if sym.search(body_l) or (company and company in body_l):
        return 0.6
    return 0.4


def backfill(
    *,
    recompute_all: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    db_path: Path = DATABASE_PATH,
) -> int:
    from src.features.news_sentiment import FinBertSentiment

    where = "" if recompute_all else "WHERE sentiment_score IS NULL OR relevance_score IS NULL"
    lim = f" LIMIT {int(limit)}" if limit else ""

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT id, ticker, title, content FROM news {where}{lim}"
    ).fetchall()
    if not rows:
        print("No rows need scoring.")
        conn.close()
        return 0

    print(f"Scoring {len(rows):,} headlines with FinBERT ...")
    scorer = FinBertSentiment(cache_db=PROCESSED_DATA_DIR / "finbert_cache.db", batch_size=32)
    # Score on title + first part of content for a richer signal.
    texts = [f"{r['title']} {r['content'] or ''}".strip()[:400] for r in rows]
    scores = scorer.score_headlines(texts)  # (N, 3) pos, neg, neu

    updates: list[tuple[float, float, int]] = []
    for r, (pos, neg, _neu) in zip(rows, scores):
        sentiment = float(pos - neg)  # signed, [-1, 1]
        relevance = _relevance(r["ticker"], r["title"], r["content"])
        updates.append((sentiment, relevance, r["id"]))

    if dry_run:
        for s, rel, _id in updates[:10]:
            print(f"  id={_id}  sentiment={s:+.3f}  relevance={rel:.2f}")
        print(f"(dry-run) would update {len(updates):,} rows")
        conn.close()
        return 0

    conn.executemany(
        "UPDATE news SET sentiment_score = ?, relevance_score = ? WHERE id = ?",
        updates,
    )
    conn.commit()

    chk = conn.execute(
        "SELECT COUNT(*) c, AVG(sentiment_score) s, AVG(relevance_score) r "
        "FROM news WHERE sentiment_score IS NOT NULL"
    ).fetchone()
    conn.close()
    print(f"Updated {len(updates):,} rows.")
    print(f"Now populated: {chk['c']:,} rows  avg_sentiment={chk['s']:+.3f}  avg_relevance={chk['r']:.3f}")
    return len(updates)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill FinBERT sentiment + relevance into news")
    ap.add_argument("--all", action="store_true", help="Recompute every row (not just NULLs)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    backfill(recompute_all=args.all, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
