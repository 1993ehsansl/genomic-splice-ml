import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from genosplice.api import app
from genosplice.synthetic import write_demo
from genosplice.data import load
from genosplice.models import PositionalMarkovClassifier
import joblib


def test_api(tmp_path, monkeypatch):
    from genosplice.api import get_model
    get_model.cache_clear()
    data = load(write_demo(tmp_path / 'd.data', n_per_class=20))
    model = PositionalMarkovClassifier().fit(data.sequences, data.labels)
    p = tmp_path / 'model.joblib'
    joblib.dump(model, p)
    monkeypatch.setenv('MODEL_PATH', str(p))
    client = TestClient(app)
    assert client.get('/health').json()['model_available'] is True
    response = client.post('/predict', json={'sequence': data.sequences[0]})
    assert response.status_code == 200
    assert abs(sum(response.json()['probabilities'].values()) - 1) < 1e-5
    assert client.post('/predict', json={'sequence': 'AA'}).status_code == 422
    get_model.cache_clear()
