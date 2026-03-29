"""Tests for the news collector's filtering and insertion logic.

These tests mock the Finnhub API to avoid network calls.
"""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from src.database.schema import DatabaseSchema
from src.data_collection.news_collector import NewsCollector


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    DatabaseSchema(db_path).create_all_tables()
    return db_path


def _make_article(headline, url, ts=1709000000, summary=""):
    return {
        "headline": headline,
        "url": url,
        "datetime": ts,
        "source": "test",
        "summary": summary,
    }


class TestNewsCollectorFilter:
    """Test the _filter_relevant method without hitting any API."""

    def test_keeps_matching_ticker_in_headline(self):
        collector = NewsCollector(api_key="fake", db_path="/dev/null")
        articles = [
            _make_article("AAPL beats earnings expectations", "https://a.com/1"),
            _make_article("Unrelated tech article", "https://a.com/2"),
        ]
        result = collector._filter_relevant("AAPL", articles)
        assert len(result) == 1
        assert "AAPL" in result[0]["headline"]

    def test_keeps_company_name_match(self):
        collector = NewsCollector(api_key="fake", db_path="/dev/null")
        articles = [
            _make_article("Tesla announces new factory plans", "https://a.com/1"),
        ]
        result = collector._filter_relevant("TSLA", articles)
        assert len(result) == 1

    def test_short_ticker_uses_word_boundary(self):
        """Short tickers like 'GS' should not match 'things' or 'gears'."""
        collector = NewsCollector(api_key="fake", db_path="/dev/null")
        articles = [
            _make_article("GS reports record revenue", "https://a.com/1"),
            _make_article("Good things happening in market", "https://a.com/2"),
            _make_article("Gears of industry turning", "https://a.com/3"),
        ]
        result = collector._filter_relevant("GS", articles)
        assert len(result) == 1
        assert "GS reports" in result[0]["headline"]

    def test_fallback_when_filter_too_aggressive(self):
        """If < 10% match, return all articles."""
        collector = NewsCollector(api_key="fake", db_path="/dev/null")
        articles = [_make_article(f"Article {i}", f"https://a.com/{i}")
                     for i in range(20)]
        result = collector._filter_relevant("AAPL", articles)
        assert len(result) == 20

    def test_checks_summary_field(self):
        """Ticker mentioned in summary (not headline) should still match."""
        collector = NewsCollector(api_key="fake", db_path="/dev/null")
        articles = [
            _make_article("Market update today", "https://a.com/1",
                          summary="AAPL shares rose 3% in trading"),
        ]
        result = collector._filter_relevant("AAPL", articles)
        assert len(result) == 1


class TestNewsCollectorInsert:
    """Test article insertion into the database."""

    def test_insert_valid_articles(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()

        articles = [
            _make_article("AAPL up today", "https://a.com/1"),
            _make_article("AAPL earnings beat", "https://a.com/2"),
        ]
        added, dupes, skipped = NewsCollector._insert_articles("AAPL", articles, cursor)
        conn.commit()
        conn.close()

        assert added == 2
        assert dupes == 0
        assert skipped == 0

    def test_skips_missing_url(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()

        articles = [{"headline": "No URL", "datetime": 1709000000, "source": "x"}]
        added, dupes, skipped = NewsCollector._insert_articles("AAPL", articles, cursor)
        conn.commit()
        conn.close()

        assert added == 0
        assert skipped == 1

    def test_skips_missing_datetime(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()

        articles = [{"headline": "Test", "url": "https://a.com/1", "source": "x"}]
        added, dupes, skipped = NewsCollector._insert_articles("AAPL", articles, cursor)
        conn.commit()
        conn.close()

        assert added == 0
        assert skipped == 1

    def test_duplicate_url_skipped(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()

        articles = [_make_article("AAPL news", "https://a.com/same")]
        added1, _, _ = NewsCollector._insert_articles("AAPL", articles, cursor)
        added2, dupes2, _ = NewsCollector._insert_articles("AAPL", articles, cursor)
        conn.commit()
        conn.close()

        assert added1 == 1
        assert added2 == 0
        assert dupes2 == 1

    def test_same_url_different_ticker_allowed(self, tmp_db):
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()

        articles = [_make_article("Market news", "https://a.com/shared")]
        a1, _, _ = NewsCollector._insert_articles("AAPL", articles, cursor)
        a2, _, _ = NewsCollector._insert_articles("TSLA", articles, cursor)
        conn.commit()
        conn.close()

        assert a1 == 1
        assert a2 == 1


class TestNewsCollectorCollect:
    """Test the full collect() method with a mocked API client."""

    def test_collect_mocked_api(self, tmp_db):
        collector = NewsCollector(api_key="fake", db_path=tmp_db)

        mock_articles = [
            _make_article("AAPL beats earnings", "https://a.com/1"),
            _make_article("AAPL launches new product", "https://a.com/2"),
        ]
        collector.client.get_company_news = MagicMock(return_value=mock_articles)

        stats = collector.collect(["AAPL"], "2026-01-01", "2026-01-31")

        assert "AAPL" in stats["tickers_succeeded"]
        assert stats["rows_added"] == 2
        assert len(stats["tickers_failed"]) == 0

    def test_collect_empty_api_response(self, tmp_db):
        collector = NewsCollector(api_key="fake", db_path=tmp_db)
        collector.client.get_company_news = MagicMock(return_value=[])

        stats = collector.collect(["AAPL"], "2026-01-01", "2026-01-31")

        assert "AAPL" in stats["tickers_succeeded"]
        assert stats["rows_added"] == 0

    def test_collect_logs_run(self, tmp_db):
        collector = NewsCollector(api_key="fake", db_path=tmp_db)
        collector.client.get_company_news = MagicMock(return_value=[])

        collector.collect(["AAPL"], "2026-01-01", "2026-01-31")

        conn = sqlite3.connect(tmp_db)
        count = conn.execute(
            "SELECT COUNT(*) FROM run_log WHERE run_type='news_collection'"
        ).fetchone()[0]
        conn.close()

        assert count >= 1
