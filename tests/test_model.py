import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.model_loader import ModelPredictor

# Chemins vers le modèle et les features
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.09

@pytest.fixture
def predictor():
    """Instanciation simple du modèle"""
    return ModelPredictor(MODEL_PATH, TOP_FEATURES_PATH, SEUIL_METIER)

@pytest.fixture
def valid_payload(predictor):
    """Payload avec toutes les features, valeurs par défaut 0.0"""
    return {feat: 0.0 for feat in predictor.top_features}

def test_predict_proba_classe(predictor, valid_payload):
    """Vérifie que predict_proba renvoie des valeurs entre 0 et 1
       et que predict_class renvoie 0 ou 1"""
    df = pd.DataFrame([valid_payload])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]
    assert 0 <= proba <= 1
    assert classe in [0, 1]

def test_feature_manquante(predictor):
    """Vérifie que l’absence d’une feature déclenche une KeyError"""
    df = pd.DataFrame([{predictor.top_features[0]: 0.0}])  # seule 1 feature
    with pytest.raises(KeyError):
        predictor.predict_proba(df)

def test_seuil_metier(predictor, valid_payload):
    """Cas limite : si la proba = seuil, la classe doit être 1"""
    df = pd.DataFrame([valid_payload])

    class DummyModel:
        """Dummy model qui renvoie exactement le seuil métier"""
        def predict_proba(self, X):
            return np.array([[1 - SEUIL_METIER, SEUIL_METIER]])

    predictor.model = DummyModel()
    assert predictor.predict_class(df)[0] == 1



