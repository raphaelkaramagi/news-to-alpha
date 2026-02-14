"""Centralized configuration for the entire project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (explicit path so it works from any cwd)
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
DATABASE_PATH = DATA_DIR / "database.db"

# Create directories if needed
for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
             RAW_DATA_DIR / "prices", RAW_DATA_DIR / "news"]:
    _dir.mkdir(parents=True, exist_ok=True)

# Tickers we're predicting
TICKERS: list[str] = [
    "AAPL", "NVDA", "WMT", "LLY", "JPM", 
    "XOM", "MCD", "TSLA", "DAL", "MAR", 
    "GS", "NFLX", "META", "ORCL", "PLTR"
]

TICKER_TO_COMPANY: dict[str, str] = {
    "AAPL": "Apple", "NVDA": "NVIDIA", "WMT": "Walmart",
    "LLY": "Eli Lilly", "JPM": "JPMorgan Chase", "XOM": "Exxon Mobil",
    "MCD": "McDonald's", "TSLA": "Tesla", "DAL": "Delta Air Lines",
    "MAR": "Marriott International", "GS": "Goldman Sachs Group",
    "NFLX": "Netflix", "META": "Meta", "ORCL": "Oracle", "PLTR": "Palantir"
}

# API keys (from .env file)
FINNHUB_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# Prediction settings
PREDICTION_TYPE: str = "binary"  # up/down
CUTOFF_TIMEZONE: str = "US/Eastern"
CUTOFF_HOUR: int = 16  # 4 PM ET market close

# Data collection defaults
DEFAULT_LOOKBACK_DAYS: int = 21
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SECONDS: float = 2.0

# Model hyperparameters (will tune later)
LSTM_CONFIG: dict = {
    "sequence_length": 60,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "lstm_units": [50, 50],
    "dropout": 0.2,
}

NLP_CONFIG: dict = {
    "max_features": 5000,
    "max_news_per_day": 20,
    "min_relevance_score": 0.3,
}
