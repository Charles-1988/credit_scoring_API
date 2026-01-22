import pandas as pd
import pytest
from src.model_loader import ModelPredictor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/best_model_lightgbm.pkl"
TOP_FEATURES_PATH = BASE_DIR / "data/top_features.csv"
SEUIL_METIER = 0.35

# Fonction helper pour créer un predictor local
def get_test_predictor():
    return ModelPredictor(
        model_path=MODEL_PATH,
        top_features_path=TOP_FEATURES_PATH,
        threshold=SEUIL_METIER
    )

def test_model_file_exists():
    predictor = get_test_predictor()
    assert predictor.model is not None

def test_top_features_file_exists():
    predictor = get_test_predictor()
    assert len(predictor.top_features) > 0

def test_predict_proba():
    predictor = get_test_predictor()
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    probas = predictor.predict_proba(test_df)
    assert all(0 <= p <= 1 for p in probas)

def test_predict_class():
    predictor = get_test_predictor()
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features}])
    proba = predictor.predict_proba(test_df)[0]
    classe = predictor.predict_class(test_df)[0]
    assert classe in [0, 1]
    assert (proba >= SEUIL_METIER) == (classe == 1)

def test_invalid_model_path():
    with pytest.raises(FileNotFoundError):
        ModelPredictor(model_path="models/inexistant.pkl", top_features_path=TOP_FEATURES_PATH)

def test_missing_feature():
    predictor = get_test_predictor()
    test_df = pd.DataFrame([{feat: 0 for feat in predictor.top_features[:-1]}])  # dernière feature manquante
    with pytest.raises(KeyError):
        predictor.predict_proba(test_df)


