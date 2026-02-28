#!/usr/bin/env python3
"""Train the NLP baseline and produce predictions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.nlp_baseline import NLPBaseline  # noqa: E402


def main() -> None:
    trainer = NLPBaseline()
    info = trainer.train()
    print(f"Trained model: {info['model_path']}")
    n = trainer.predict_and_save()
    print(f"Saved and inserted {n} predictions")


if __name__ == "__main__":
    main()
