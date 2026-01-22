import pandas as pd
import pytest
from src.model_loader import ModelPredictor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.35

predictor = ModelPredictor(
    model_path=MODEL_PATH,
    top_features_path=TOP_FEATURES_PATH,
    threshold=SEUIL_METIER
)
def test_model_file_exists():
    """Vérifie que le modèle est chargé."""
    assert predictor.model is not None

def test_top_features_file_exists():
    """Vérifie que la liste des features n’est pas vide."""
    assert len(predictor.top_features) > 0

def test_predict_proba():
    """Teste que la prédiction de probabilité retourne des valeurs entre 0 et 1."""
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    probas = predictor.predict_proba(test_df)
    assert all(0 <= p <= 1 for p in probas)

def test_predict_class():
    """Teste que la prédiction de classe retourne 0 ou 1 et respecte le seuil métier."""
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    proba = predictor.predict_proba(test_df)[0]
    classe = predictor.predict_class(test_df)[0]
    assert classe in [0, 1]
    assert (proba >= SEUIL_METIER) == (classe == 1)

def test_invalid_model_path():
    """Vérifie que le code lève une erreur si le modèle n'existe pas."""
    with pytest.raises(FileNotFoundError):
        ModelPredictor(model_path="models/inexistant.pkl", top_features_path=TOP_FEATURES_PATH)

def test_missing_feature():
    """Vérifie que le code lève une erreur si une feature est manquante."""
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features[:-1]}])  # dernière feature manquante
    with pytest.raises(KeyError):
        predictor.predict_proba(test_df)

