#!/usr/bin/env python3
"""
Create (or update) the SQLite database with all project tables.

Usage:
    python scripts/setup_database.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.schema import DatabaseSchema  # noqa: E402


def main() -> None:
    print("Setting up database ...")
    schema = DatabaseSchema()
    schema.create_all_tables()
    print(f"\nDatabase ready at {schema.db_path.resolve()}")
    print(
        "Tables: prices, news, labels, predictions, run_log\n"
        "\nNext steps:\n"
        "  python scripts/collect_prices.py\n"
        "  python scripts/collect_news.py"
    )


if __name__ == "__main__":
    main()
