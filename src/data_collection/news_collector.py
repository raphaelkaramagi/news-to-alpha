"""Collect news articles from Finnhub API."""

import json
import sqlite3
import logging
from datetime import datetime, timedelta

import pytz

from src.data_collection.base_collector import BaseCollector
from src.utils.api_clients import FinnhubClient
from src.config import DATABASE_PATH, FINNHUB_API_KEY, TICKER_TO_COMPANY

logger = logging.getLogger(__name__)


class NewsCollector(BaseCollector):
    def __init__(self, api_key: str = FINNHUB_API_KEY, db_path: str = str(DATABASE_PATH)):
        super().__init__(db_path)
        self.client = FinnhubClient(api_key)

    def collect(self, tickers: list[str], start_date: str, end_date: str, **kwargs) -> dict:
        """Fetch news for tickers and insert into database."""
        stats = {
            "tickers_succeeded": [],
            "tickers_failed": [],
            "rows_added": 0,
            "duplicates_skipped": 0,
            "errors": {},
        }

        run_start = datetime.now()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for ticker in tickers:
            try:
                self.logger.info("Fetching news for %s ...", ticker)
                articles = self.client.get_company_news(ticker, start_date, end_date)

                if not articles:
                    self.logger.warning("No articles for %s", ticker)
                    stats["tickers_succeeded"].append(ticker)  # not a failure
                    continue

                # Keep only articles that mention the ticker or company name
                relevant = self._filter_relevant(ticker, articles)
                self.logger.info("%s: %d / %d articles relevant", ticker, len(relevant), len(articles))

                added, dupes = self._insert_articles(ticker, relevant, cursor)
                stats["rows_added"] += added
                stats["duplicates_skipped"] += dupes
                stats["tickers_succeeded"].append(ticker)

                self.logger.info("%s: +%d articles, %d duplicates", ticker, added, dupes)

            except Exception as exc:
                self.logger.error("%s failed: %s", ticker, exc)
                stats["tickers_failed"].append(ticker)
                stats["errors"][ticker] = str(exc)

        conn.commit()
        conn.close()

        duration = (datetime.now() - run_start).total_seconds()
        self._log_run(stats, run_start, duration)
        return stats

    def _filter_relevant(self, ticker: str, articles: list[dict]) -> list[dict]:
        """Keep articles where headline mentions ticker or company name."""
        company = TICKER_TO_COMPANY.get(ticker, "").lower()
        result = []
        for art in articles:
            headline = art.get("headline", "").lower()
            if ticker.lower() in headline or (company and company in headline):
                result.append(art)

        # If too aggressive (< 10% kept), fall back to all
        if len(result) < max(1, len(articles) * 0.10):
            return articles
        return result

    @staticmethod
    def _insert_articles(ticker: str, articles: list[dict], cursor: sqlite3.Cursor) -> tuple[int, int]:
        """Insert articles into news table."""
        added = 0
        dupes = 0

        for art in articles:
            try:
                pub_ts = art.get("datetime", 0)
                pub_dt = datetime.fromtimestamp(pub_ts, tz=pytz.UTC)
                pub_et = pub_dt.astimezone(pytz.timezone("US/Eastern"))

                cursor.execute(
                    """INSERT INTO news (url, ticker, title, source, published_at, content)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (art.get("url", ""), ticker, art.get("headline", ""), 
                     art.get("source", ""), pub_et.isoformat(), art.get("summary", ""))
                )
                added += 1
            except sqlite3.IntegrityError:
                dupes += 1
            except Exception as exc:
                logger.debug("Insert error: %s", exc)

        return added, dupes

    def _log_run(self, stats: dict, started: datetime, duration: float) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO run_log 
               (run_type, status, tickers_attempted, tickers_succeeded, tickers_failed, 
                rows_added, duplicates_skipped, error_message, started_at, completed_at, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("news_collection", "success" if not stats["tickers_failed"] else "partial",
             json.dumps(stats["tickers_succeeded"] + stats["tickers_failed"]),
             json.dumps(stats["tickers_succeeded"]), json.dumps(stats["tickers_failed"]),
             stats["rows_added"], stats["duplicates_skipped"],
             json.dumps(stats["errors"]) if stats["errors"] else None,
             started.isoformat(), datetime.now().isoformat(), round(duration, 2))
        )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.database.schema import DatabaseSchema
    DatabaseSchema().create_all_tables()

    collector = NewsCollector()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"\nTest: collecting AAPL news  {start} -> {end}\n")
    result = collector.collect(["AAPL"], start, end)
    print(f"\nArticles: {result['rows_added']}, Duplicates: {result['duplicates_skipped']}")
