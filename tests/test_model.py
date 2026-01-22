import pytest
import pandas as pd
from pathlib import Path
from src.model_loader import ModelPredictor

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

def test_proba_classe_valide(predictor, valid_payload):
    """Vérifie que proba est entre 0 et 1 et classe = 0 ou 1"""
    df = pd.DataFrame([valid_payload])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]
    assert 0 <= proba <= 1
    assert classe in [0, 1]

def test_feature_manquante(predictor):
    """Vérifie que l’absence d’une feature déclenche KeyError"""
    df = pd.DataFrame([{predictor.top_features[0]: 0.0}])
    with pytest.raises(KeyError):
        predictor.predict_proba(df)

def test_seuil_metier(predictor, valid_payload):
    """Cas limite : si la proba = seuil, classe = 1"""
    df = pd.DataFrame([valid_payload])
    # On force la proba à être exactement le seuil
    predictor.model.predict_proba = lambda X: [[1 - SEUIL_METIER, SEUIL_METIER]]
    assert predictor.predict_class(df)[0] == 1


