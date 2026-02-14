"""Abstract base class for data collectors."""

from abc import ABC, abstractmethod
import logging


class BaseCollector(ABC):
    def __init__(self, db_path: str = "data/database.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def collect(self, tickers: list[str], start_date: str, end_date: str, **kwargs) -> dict:
        """Collect data for tickers in date range. Returns stats dict."""
