"""Top-N publisher one-hot encoder for news rows.

Given a training set's `sources` JSON column (list of source strings per
ticker-day), we pick the top-N most frequent publishers and create a
compact fixed-width feature vector:

    [pub_1_present, pub_2_present, ..., pub_N_present, pub_other_present]

A ticker-day gets a 1 for every publisher on its list, 0 otherwise.
`pub_other_present` flips to 1 when any non-top-N publisher is present.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Iterable


def _parse_sources(cell: str | list[str] | None) -> list[str]:
    if cell is None or isinstance(cell, float):
        return []
    if isinstance(cell, list):
        return [str(s).strip() for s in cell if s]
    try:
        return [str(s).strip() for s in json.loads(cell) if s]
    except (ValueError, TypeError):
        return []


class PublisherOneHot:
    """One-hot encoder for publisher lists.

    Usage
    -----
    enc = PublisherOneHot(top_n=15)
    enc.fit(train_df["sources"])
    X = enc.transform(df["sources"])    # shape (n_rows, top_n + 1)
    """

    def __init__(self, top_n: int = 15) -> None:
        self.top_n = top_n
        self.top_publishers: list[str] = []

    @property
    def vocab_size(self) -> int:
        return len(self.top_publishers) + 1  # +1 for "other"

    @property
    def feature_names(self) -> list[str]:
        return [f"pub__{p}" for p in self.top_publishers] + ["pub__other"]

    def fit(self, sources: Iterable[str | list[str] | None]) -> "PublisherOneHot":
        counter: Counter[str] = Counter()
        for cell in sources:
            for s in _parse_sources(cell):
                counter[s] += 1
        self.top_publishers = [name for name, _ in counter.most_common(self.top_n)]
        return self

    def transform(self, sources: Iterable[str | list[str] | None]) -> "np.ndarray":  # noqa: F821
        import numpy as np

        index = {p: i for i, p in enumerate(self.top_publishers)}
        rows = list(sources)
        out = np.zeros((len(rows), self.vocab_size), dtype=np.float32)
        other_idx = len(self.top_publishers)
        for i, cell in enumerate(rows):
            pubs = _parse_sources(cell)
            saw_other = False
            for p in pubs:
                if p in index:
                    out[i, index[p]] = 1.0
                else:
                    saw_other = True
            if saw_other:
                out[i, other_idx] = 1.0
        return out

    def fit_transform(self, sources):
        self.fit(sources)
        return self.transform(sources)
