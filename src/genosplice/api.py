"""Validated inference API. Load the trained model from MODEL_PATH."""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from .data import ALLOWED

app = FastAPI(title='GenomicSpliceML', version='0.1.0', description='Research demonstration, not for clinical use')

class PredictionRequest(BaseModel):
    sequence: str = Field(description='A 60-base DNA window (IUPAC ambiguous bases accepted)')

    @field_validator('sequence')
    @classmethod
    def valid_sequence(cls, value: str) -> str:
        s = ''.join(value.upper().split())
        if len(s) != 60 or set(s) - ALLOWED:
            raise ValueError('DNA sequence must be exactly 60 valid IUPAC bases')
        return s

@lru_cache(maxsize=1)
def get_model():
    path = Path(os.environ.get('MODEL_PATH', 'artifacts/model.joblib'))
    if not path.exists():
        raise FileNotFoundError(f'Missing trained artifact: {path}')
    # Loading joblib/pickle files is unsafe from untrusted sources. Use only YOUR trained artifacts.
    return joblib.load(path)

@app.get('/health')
def health():
    model_path = Path(os.environ.get('MODEL_PATH', 'artifacts/model.joblib'))
    return {'status': 'ok' if model_path.exists() else 'model_not_found', 'model_available': model_path.exists()}

@app.post('/predict')
def predict(request: PredictionRequest):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    p = model.predict_proba([request.sequence])[0]
    classes = model.classes_.tolist()
    idx = int(p.argmax())
    return {'class': classes[idx], 'description': {'EI': 'donor junction', 'IE': 'acceptor junction', 'N': 'neither'}[classes[idx]],
            'probabilities': {c: round(float(prob), 6) for c, prob in zip(classes, p)},
            'disclaimer': 'Research-only: not calibrated for clinical or diagnostic decisions.'}
