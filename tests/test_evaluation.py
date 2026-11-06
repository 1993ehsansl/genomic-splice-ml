import numpy as np
from genosplice.evaluation import metrics, bootstrap_macro_f1, expected_calibration_error


def test_perfect_metrics():
    labels = np.array(['EI', 'IE', 'N'])
    p = np.eye(3)
    result = metrics(labels, p, labels)
    assert result['accuracy'] == 1
    assert result['macro_f1'] == 1
    assert result['macro_auc_ovr'] == 1
    assert result['confusion_matrix'] == [[1,0,0], [0,1,0], [0,0,1]]
    assert expected_calibration_error(labels, labels, p) == 0
    confidence = bootstrap_macro_f1(labels, p, labels, n_boot=30)
    assert confidence['95pct_ci_macro_f1'][1] == 1
