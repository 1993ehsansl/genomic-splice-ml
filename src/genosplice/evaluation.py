"""Auditable metrics, uncertainty, model diagnostics, and figures."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve


def expected_calibration_error(y, pred, proba, bins=10):
    confidence = proba.max(axis=1)
    correct = (np.asarray(y) == np.asarray(pred)).astype(float)
    total = 0.
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= lo) & (confidence <= hi if hi == 1 else confidence < hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return total


def metrics(y_true, proba, classes) -> dict:
    y = np.asarray(y_true)
    p = np.asarray(proba)
    classes = np.asarray(classes)
    pred = classes[p.argmax(axis=1)]
    output = {
        'n': len(y), 'accuracy': round(float(accuracy_score(y, pred)), 6),
        'macro_f1': round(float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)), 6),
        'balanced_accuracy': round(float(balanced_accuracy_score(y, pred)), 6),
        'log_loss': round(float(log_loss(y, p, labels=classes)), 6),
        'multiclass_brier': round(float(np.mean(np.sum((p - (y[:, None] == classes).astype(float))**2, axis=1))), 6),
        'ece_10bins': round(float(expected_calibration_error(y, pred, p)), 6),
        'confusion_matrix': confusion_matrix(y, pred, labels=classes).tolist(),
        'classes': classes.tolist(),
    }
    try:
        if len(classes) == 3 and len(set(y)) == 3:
            output['macro_auc_ovr'] = round(float(roc_auc_score(y, p, labels=classes, multi_class='ovr', average='macro')), 6)
        else:
            output['macro_auc_ovr'] = None
    except ValueError:
        output['macro_auc_ovr'] = None
    return output


def bootstrap_macro_f1(y_true, proba, classes, n_boot=300, seed=42):
    y = np.asarray(y_true)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[idx], pred[idx], labels=classes, average='macro', zero_division=0))
    low, high = np.quantile(values, [0.025, .975])
    return {'method': 'ordinary_noncluster_bootstrap', 'resamples': n_boot,
            '95pct_ci_macro_f1': [round(float(low), 6), round(float(high), 6)]}


def save_plots(results: dict, report: dict, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    out = Path(out_dir)
    names = list(results)
    scores = [results[n]['macro_f1'] for n in names]
    plt.figure(figsize=(8, 4))
    plt.barh(names, scores)
    plt.xlim(0, 1)
    plt.xlabel('Validation macro-F1 (higher is better)')
    plt.tight_layout()
    plt.savefig(out / 'model_comparison.png', dpi=160)
    plt.close()
    mat = np.asarray(report['confusion_matrix'])
    plt.figure(figsize=(5, 4))
    plt.imshow(mat, cmap='Blues')
    ticks = list(range(len(report['classes'])))
    plt.xticks(ticks, report['classes'])
    plt.yticks(ticks, report['classes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for row in range(len(ticks)):
        for col in range(len(ticks)):
            plt.text(col, row, str(mat[row, col]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out / 'test_confusion_matrix.png', dpi=160)
    plt.close()
