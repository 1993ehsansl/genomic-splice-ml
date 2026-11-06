from pathlib import Path
import pytest
from genosplice.data import load, data_report
from genosplice.synthetic import write_demo


def test_demo_parsing(tmp_path):
    ds = load(write_demo(tmp_path / 'demo.data', n_per_class=12))
    assert ds.sequences.shape == (36,)
    assert set(ds.labels) == {'EI', 'IE', 'N'}
    assert data_report(ds)['n_samples'] == 36


def test_invalid_input(tmp_path):
    p = tmp_path / 'bad.data'
    p.write_text('EI,example,AAGT\n')
    with pytest.raises(ValueError, match='invalid record'):
        load(p)
