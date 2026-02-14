"""Finnhub API wrapper with rate-limiting."""

import time
import logging
from datetime import datetime

import requests
import pytz

logger = logging.getLogger(__name__)


class FinnhubClient:
    """Wrapper for Finnhub REST API. Free tier = 60 calls/min."""

    BASE_URL = "https://finnhub.io/api/v1"
    MAX_CALLS_PER_MINUTE = 58  # buffer below hard 60 limit

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Finnhub API key is empty. Set NEWS_API_KEY in your .env file.")
        self.api_key = api_key
        self._calls = 0
        self._window_start = time.monotonic()

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._window_start
        if elapsed > 60:
            self._calls = 0
            self._window_start = time.monotonic()

        self._calls += 1
        if self._calls >= self.MAX_CALLS_PER_MINUTE:
            wait = 60 - elapsed
            if wait > 0:
                logger.info("Rate-limit pause: %.1f s", wait)
                time.sleep(wait)
            self._calls = 0
            self._window_start = time.monotonic()

    def get_company_news(self, ticker: str, start_date: str, end_date: str) -> list[dict]:
        """Fetch news articles for a ticker. Dates are YYYY-MM-DD strings."""
        self._rate_limit()
        try:
            resp = requests.get(
                f"{self.BASE_URL}/company-news",
                params={"symbol": ticker, "from": start_date, "to": end_date, "token": self.api_key},
                timeout=15,
            )
            resp.raise_for_status()
            articles = resp.json()
            logger.info("Fetched %d articles for %s", len(articles), ticker)
            return articles
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching news for %s", ticker)
            return []
        except requests.exceptions.RequestException as exc:
            logger.error("Error fetching news for %s: %s", ticker, exc)
            return []

    @staticmethod
    def filter_by_cutoff(articles: list[dict], cutoff_hour: int = 16, timezone_str: str = "US/Eastern") -> list[dict]:
        """Keep only articles published before cutoff_hour in ET."""
        tz = pytz.timezone(timezone_str)
        return [a for a in articles if datetime.fromtimestamp(a["datetime"], tz=tz).hour < cutoff_hour]
