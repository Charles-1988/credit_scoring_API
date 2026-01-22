# tests/test_model.py
import pandas as pd
from fastapi.testclient import TestClient
from pathlib import Path

from src.model_loader import ModelPredictor
from src.main import app

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.09

client = TestClient(app)

def get_predictor():
    return ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)

def get_valid_payload():
    predictor = get_predictor()
    return {feat: 0.0 for feat in predictor.top_features}


def test_predictor_outputs():
    predictor = get_predictor()
    df = pd.DataFrame([get_valid_payload()])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]
    assert 0 <= proba <= 1
    assert classe in [0, 1]

def test_missing_feature_raises():
    predictor = get_predictor()
    payload = get_valid_payload()
    payload.pop(next(iter(payload)))  # enlever une feature
    df = pd.DataFrame([payload])
    try:
        predictor.predict_proba(df)
        assert False
    except KeyError:
        pass

def test_api_predict_success():
    response = client.post("/predict", json=get_valid_payload())
    assert response.status_code == 200
    data = response.json()
    assert "proba" in data and "classe" in data

def test_api_predict_missing_feature():
    predictor = get_predictor()
    payload = {predictor.top_features[0]: 0.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic gère la validation
