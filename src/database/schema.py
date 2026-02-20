"""Database schema - 5 tables for prices, news, labels, predictions, logs."""

import sqlite3
import logging
from pathlib import Path
from src.config import DATABASE_PATH

logger = logging.getLogger(__name__)


class DatabaseSchema:
    def __init__(self, db_path: str | Path = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def create_all_tables(self) -> None:
        """Create all 5 tables (idempotent - won't break if already exists)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            self._create_prices_table(cursor)
            self._create_news_table(cursor)
            self._create_labels_table(cursor)
            self._create_predictions_table(cursor)
            self._create_run_log_table(cursor)
            conn.commit()
            logger.info("All database tables created successfully")
        except Exception as exc:
            conn.rollback()
            logger.error("Database creation failed: %s", exc)
            raise
        finally:
            conn.close()

    def _create_prices_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                date            TEXT    NOT NULL,
                open            REAL,
                high            REAL,
                low             REAL,
                close           REAL    NOT NULL,
                volume          INTEGER,
                adjusted_close  REAL,
                created_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date)")

    def _create_news_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                url             TEXT    NOT NULL,
                ticker          TEXT    NOT NULL,
                title           TEXT    NOT NULL,
                source          TEXT,
                published_at    TEXT    NOT NULL,
                content         TEXT,
                sentiment_score REAL,
                relevance_score REAL,
                created_at      TEXT DEFAULT (datetime('now')),
                CHECK(relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1)),
                UNIQUE(url, ticker)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker_published ON news(ticker, published_at)")
        
    def _create_labels_table(self, cursor: sqlite3.Cursor) -> None:
        """Ground truth: did stock go up or down next day?"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL,
                date            TEXT    NOT NULL,
                label_binary    INTEGER,
                label_return    REAL,
                close_t         REAL,
                close_t_plus_1  REAL,
                created_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_ticker_date ON labels(ticker, date)")

    def _create_predictions_table(self, cursor: sqlite3.Cursor) -> None:
        """Store model predictions for evaluation."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker                  TEXT NOT NULL,
                date                    TEXT NOT NULL,
                financial_pred_proba    REAL,
                financial_confidence    REAL,
                news_pred_proba         REAL,
                news_confidence         REAL,
                news_top_headlines      TEXT,
                ensemble_pred_proba     REAL,
                ensemble_confidence     REAL,
                financial_pred_binary   INTEGER,
                news_pred_binary        INTEGER,
                ensemble_pred_binary    INTEGER,
                actual_binary           INTEGER,
                actual_return           REAL,
                model_version           TEXT,
                created_at              TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, date, model_version)
            )
        """)

    def _create_run_log_table(self, cursor: sqlite3.Cursor) -> None:
        """Track when data collection / training runs happened."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_log (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type            TEXT NOT NULL,
                status              TEXT NOT NULL,
                tickers_attempted   TEXT,
                tickers_succeeded   TEXT,
                tickers_failed      TEXT,
                rows_added          INTEGER,
                duplicates_skipped  INTEGER,
                error_message       TEXT,
                started_at          TEXT,
                completed_at        TEXT,
                duration_seconds    REAL
            )
        """)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    schema = DatabaseSchema()
    schema.create_all_tables()
    print(f"\nDatabase created at {schema.db_path.resolve()}")
