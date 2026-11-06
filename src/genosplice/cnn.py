"""Optional small sequence CNN, trained with early stopping on validation macro-F1."""
from __future__ import annotations
import copy
import random
import numpy as np
from .features import to_codes


def require_torch():
    try:
        import torch
        from torch import nn
        return torch, nn
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies with: pip install -e ".[deep]"') from exc


def make_network(length=60):
    torch, nn = require_torch()
    return nn.Sequential(
        nn.Conv1d(5, 32, kernel_size=7, padding=3), nn.ReLU(),
        nn.Conv1d(32, 48, kernel_size=5, padding=2), nn.ReLU(),
        nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        nn.Dropout(.25), nn.Linear(48, 3),
    )


def one_hot_tensor(sequences):
    torch, _ = require_torch()
    c = torch.as_tensor(to_codes(sequences), dtype=torch.int64)
    return torch.nn.functional.one_hot(c, num_classes=5).float().transpose(1, 2)


class DNAConvClassifier:
    """CPU-compatible 1D CNN; validation controls early stopping, test remains untouched."""
    def __init__(self, seed=42, epochs=20, patience=5, batch_size=64, lr=0.002):
        self.seed, self.epochs, self.patience = seed, epochs, patience
        self.batch_size, self.lr = batch_size, lr

    def fit(self, X, y, X_val, y_val):
        torch, nn = require_torch()
        from sklearn.metrics import f1_score
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        self.classes_ = np.array(sorted(set(y)))
        if len(self.classes_) != 3:
            raise ValueError('Expected 3-class training data')
        self.network_ = make_network()
        features = one_hot_tensor(X)
        labels = torch.tensor([np.where(self.classes_ == lab)[0][0] for lab in y], dtype=torch.int64)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.lr, weight_decay=.01)
        loss_fn = nn.CrossEntropyLoss()
        best, remaining, best_state = -1.0, self.patience, None
        rng = np.random.default_rng(self.seed)
        self.history_ = []
        for epoch in range(self.epochs):
            self.network_.train()
            perm = rng.permutation(len(y))
            for batch in np.array_split(perm, max(1, int(np.ceil(len(y) / self.batch_size)))):
                optimizer.zero_grad()
                loss = loss_fn(self.network_(features[batch]), labels[batch])
                loss.backward()
                optimizer.step()
            prob = self.predict_proba(X_val)
            pred = self.classes_[prob.argmax(axis=1)]
            score = f1_score(y_val, pred, labels=self.classes_, average='macro', zero_division=0)
            self.history_.append({'epoch': epoch + 1, 'validation_macro_f1': float(score)})
            if score > best + 1e-5:
                best, remaining = score, self.patience
                best_state = copy.deepcopy(self.network_.state_dict())
            else:
                remaining -= 1
                if remaining == 0:
                    break
        self.network_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        torch, _ = require_torch()
        self.network_.eval()
        with torch.no_grad():
            outputs = []
            for start in range(0, len(X), 256):
                logits = self.network_(one_hot_tensor(X[start:start+256]))
                outputs.append(torch.softmax(logits, dim=1).numpy())
            return np.vstack(outputs)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
