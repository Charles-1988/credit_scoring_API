# tests/test_simple.py
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from pathlib import Path
from src.model_loader import ModelPredictor
from src.main import app

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.09


@pytest.fixture
def predictor():
    return ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)

@pytest.fixture
def valid_payload(predictor):
    return {feat: 0.0 for feat in predictor.top_features}

@pytest.fixture
def client():
    return TestClient(app)


def test_model_predict(predictor, valid_payload):
    df = pd.DataFrame([valid_payload])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]
    assert 0 <= proba <= 1
    assert classe in [0, 1]

def test_model_missing_feature(predictor):
    df = pd.DataFrame([{predictor.top_features[0]: 0.0}])
    with pytest.raises(KeyError):
        predictor.predict_proba(df)


def test_api_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "API Credit Scoring active" in response.json()["message"]

def test_api_predict_valid(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["proba"] <= 1
    assert data["classe"] in [0, 1]

def test_api_predict_missing_feature(client, valid_payload):
    payload = valid_payload.copy()
    payload.pop(list(payload.keys())[0])
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  

def test_api_predict_wrong_type(client, valid_payload):
    payload = valid_payload.copy()
    payload[list(payload.keys())[0]] = "wrong_type"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  




