"""DNA sequence featurization with unknown bases handled explicitly."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

BASES = 'ACGT'


def to_codes(sequences: list[str] | np.ndarray) -> np.ndarray:
    seq = list(sequences)
    if not seq:
        return np.empty((0, 60), dtype=np.int64)
    n = len(seq[0])
    if any(len(s) != n for s in seq):
        raise ValueError('Expected fixed-length DNA strings')
    codes = np.full((len(seq), n), 4, dtype=np.int64)
    mapping = {base: i for i, base in enumerate(BASES)}
    for row, s in enumerate(seq):
        for pos, char in enumerate(s):
            codes[row, pos] = mapping.get(char, 4)
    return codes


class PositionOneHot(BaseEstimator, TransformerMixin):
    """Position-specific nucleotide encoding (five states: A,C,G,T,ambiguous)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        c = to_codes(X)
        return np.eye(5, dtype=np.float32)[c].reshape(len(c), -1)


def motif_frequencies(sequences: np.ndarray, labels: np.ndarray, window: range = range(23, 37)) -> dict:
    """Training-only positional nucleotide frequencies for interpretability."""
    out = {}
    for label in ('EI', 'IE', 'N'):
        xs = sequences[labels == label]
        if len(xs) == 0:
            continue
        freq = []
        for p in window:
            freq.append({b: float(sum(s[p] == b for s in xs) / len(xs)) for b in BASES})
        out[label] = {'positions_zero_based': list(window), 'frequencies': freq}
    return out
