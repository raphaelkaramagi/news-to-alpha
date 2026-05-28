"""Centralized configuration for the entire project.

Path layout
-----------
Local dev defaults to ``repo/data/``. Railway inference sets env vars so
everything reads from the mounted volume at ``/data`` (see Dockerfile.inference).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root (explicit path so it works from any cwd)
load_dotenv(PROJECT_ROOT / ".env")


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    return Path(raw) if raw else default


# Local dev: repo/data. Railway: set DATA_DIR=/data (or individual path env vars).
DATA_DIR = _env_path("DATA_DIR", PROJECT_ROOT / "data")
RAW_DATA_DIR = _env_path("RAW_DATA_DIR", DATA_DIR / "raw")
PROCESSED_DATA_DIR = _env_path("PROCESSED_DATA_DIR", DATA_DIR / "processed")
MODELS_DIR = _env_path("MODELS_DIR", DATA_DIR / "models")
DATABASE_PATH = _env_path("DATABASE_PATH", DATA_DIR / "database.db")

# Create directories if needed
for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
             RAW_DATA_DIR / "prices", RAW_DATA_DIR / "news"]:
    _dir.mkdir(parents=True, exist_ok=True)

# Tickers we're predicting
TICKERS: list[str] = [
    "AAPL", "NVDA", "WMT", "LLY", "JPM",
    "XOM", "MCD", "TSLA", "DAL", "MAR",
    "GS", "NFLX", "META", "ORCL", "PLTR",
    "GOOGL", "MSFT", "MU", "AMD", "AMZN",
]

TICKER_TO_COMPANY: dict[str, str] = {
    "AAPL": "Apple", "NVDA": "NVIDIA", "WMT": "Walmart",
    "LLY": "Eli Lilly", "JPM": "JPMorgan Chase", "XOM": "Exxon Mobil",
    "MCD": "McDonald's", "TSLA": "Tesla", "DAL": "Delta Air Lines",
    "MAR": "Marriott International", "GS": "Goldman Sachs Group",
    "NFLX": "Netflix", "META": "Meta", "ORCL": "Oracle", "PLTR": "Palantir",
    "GOOGL": "Alphabet", "MSFT": "Microsoft", "MU": "Micron Technology",
    "AMD": "Advanced Micro Devices", "AMZN": "Amazon",
}

# API keys (from .env file)
FINNHUB_API_KEY: str = os.getenv("NEWS_API_KEY", "")

# Prediction settings
PREDICTION_TYPE: str = "binary"  # up/down
CUTOFF_TIMEZONE: str = "US/Eastern"
CUTOFF_HOUR: int = 16  # 4 PM ET market close

# Data collection defaults
DEFAULT_LOOKBACK_DAYS: int = 365
MAX_RETRIES: int = 3
RETRY_BASE_DELAY_SECONDS: float = 2.0

# Reproducibility
RANDOM_SEED: int = 42

# Market regime feature ticker (collected alongside the stock universe).
MARKET_INDEX_TICKER: str = "SPY"

# Model hyperparameters
LSTM_CONFIG: dict = {
    "sequence_length": 60,
    "batch_size": 64,
    "epochs": 80,
    "learning_rate": 0.0005,
    "lstm_units": [64, 64],
    "dropout": 0.4,
    "weight_decay": 1e-4,
    "ticker_embed_dim": 4,
    "use_focal_loss": False,
    "focal_gamma": 2.0,
    "focal_alpha": 0.5,
    "patience": 20,
    "early_stop_metric": "auc",
    "seeds": [42, 1337, 2024],
}

NLP_CONFIG: dict = {
    "max_features": 5000,
    "max_news_per_day": 20,
    "min_relevance_score": 0.3,
}

ENSEMBLE_CONFIG: dict = {
    "meta_C": 1.0,
    "calibration_method": "isotonic",
}
