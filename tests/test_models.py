import numpy as np
from genosplice.features import PositionOneHot, to_codes
from genosplice.models import PositionalMarkovClassifier, model_candidates
from genosplice.synthetic import write_demo
from genosplice.data import load
from genosplice.train import grouped_split


def test_onehot_and_ambiguity():
    x = ['ACGT' + 'N'*56]
    arr = to_codes(x)
    assert arr[0, :5].tolist() == [0, 1, 2, 3, 4]
    assert PositionOneHot().transform(x).shape == (1, 300)


def test_probabilities_and_leakage(tmp_path):
    d = load(write_demo(tmp_path / 'd.data', n_per_class=35))
    tr, va, te = grouped_split(d.sequences, d.labels)
    assert len(set(d.sequences[tr]) & set(d.sequences[va])) == 0
    assert len(set(d.sequences[tr]) & set(d.sequences[te])) == 0
    assert len(set(d.sequences[va]) & set(d.sequences[te])) == 0
    model = PositionalMarkovClassifier(alpha=2).fit(d.sequences[tr], d.labels[tr])
    p = model.predict_proba(d.sequences[te]); assert p.shape[1] == 3
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-8)


def test_all_models_fit(tmp_path):
    d = load(write_demo(tmp_path / 'd.data', n_per_class=25))
    tr, va, _ = grouped_split(d.sequences, d.labels)
    for m in model_candidates().values():
        m.fit(d.sequences[tr], d.labels[tr])
        p = m.predict_proba(d.sequences[va])
        assert p.shape == (len(va), 3)
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
