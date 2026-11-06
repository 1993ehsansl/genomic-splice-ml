"""Interpretable and statistical benchmarks; pipeline fit only on training data."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .features import PositionOneHot, to_codes


class PositionalMarkovClassifier(BaseEstimator, ClassifierMixin):
    """Class-conditional, position-dependent 1st-order Markov model.

    It is a generative probabilistic baseline, NOT an HMM. Additive smoothing
    prevents zero likelihood for rare transitions. Ambiguity is a fifth state.
    """
    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def fit(self, X, y):
        arr = to_codes(X)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = np.log(counts / counts.sum())
        n, length = arr.shape
        self.length_ = length
        self.start_ = np.zeros((len(self.classes_), 5))
        self.transitions_ = np.zeros((len(self.classes_), length - 1, 5, 5))
        for i, cls in enumerate(self.classes_):
            subset = arr[y == cls]
            start = np.bincount(subset[:, 0], minlength=5) + self.alpha
            self.start_[i] = np.log(start / start.sum())
            for p in range(length - 1):
                mat = np.full((5, 5), self.alpha, dtype=float)
                np.add.at(mat, (subset[:, p], subset[:, p + 1]), 1)
                self.transitions_[i, p] = np.log(mat / mat.sum(axis=1, keepdims=True))
        return self

    def predict_log_proba(self, X):
        a = to_codes(X)
        if a.shape[1] != self.length_:
            raise ValueError('DNA sequence length mismatch')
        scores = np.zeros((len(a), len(self.classes_)))
        for k in range(len(self.classes_)):
            scores[:, k] = self.priors_[k] + self.start_[k, a[:, 0]]
            for p in range(self.length_ - 1):
                scores[:, k] += self.transitions_[k, p, a[:, p], a[:, p + 1]]
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


def model_candidates(seed: int = 42) -> dict:
    return {
        'prior_baseline': Pipeline([('onehot', PositionOneHot()), ('model', DummyClassifier(strategy='prior'))]),
        'positional_markov': PositionalMarkovClassifier(alpha=2.0),
        'position_logreg': Pipeline([('onehot', PositionOneHot()), ('model', LogisticRegression(C=1.0, max_iter=1600, class_weight='balanced', random_state=seed))]),
        'kmer_logreg': Pipeline([('kmers', CountVectorizer(analyzer='char', ngram_range=(3, 5), lowercase=False)), ('model', LogisticRegression(C=0.5, max_iter=1600, class_weight='balanced', random_state=seed))]),
    }
