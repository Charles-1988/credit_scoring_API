import pytest
import pandas as pd
from unittest.mock import Mock
from pathlib import Path
from src.model_loader import ModelPredictor
from src.main import predict_logic, credit_decision

# Tests ModelPredictor
def test_predict_proba_missing_feature():
    "vérifie si une feature obligatoire est absente"
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.top_features = ["a", "b"]
    predictor.model = Mock()
    df = pd.DataFrame([{"a": 1}])
    with pytest.raises(KeyError):
        predictor.predict_proba(df)

def test_predict_proba_calls_model():
    "on vérifie que le modéle retourne la bonne probabilité"
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.top_features = ["a", "b"]
    mock_model = Mock()
    mock_model.predict_proba.return_value = [[0.7, 0.3]]
    predictor.model = mock_model
    df = pd.DataFrame([{"a": 1, "b": 2}])
    proba = predictor.predict_proba(df)
    assert proba[0] == 0.3
    mock_model.predict_proba.assert_called_once()

def test_predict_class_threshold():
    "vérifie que predict class convertit bien les probabiltés en 0/1"
    predictor = ModelPredictor.__new__(ModelPredictor)
    predictor.threshold = 0.5
    predictor.predict_proba = Mock(return_value=pd.Series([0.4, 0.6]))
    df = pd.DataFrame([{}, {}])
    classes = predictor.predict_class(df)
    assert list(classes) == [0, 1]

def test_invalid_model_path():
    "vérifie si le fichier du modéle n'existe pas "
    with pytest.raises(FileNotFoundError):
        ModelPredictor(model_path=Path("chemin/inexistant.pkl"))

# Tests logique API
def test_predict_logic_ok():
    "vérifue que predict proba retourne la proba et la classe  "
    mock_predictor = Mock()
    mock_predictor.predict_proba.return_value = [0.2]
    mock_predictor.predict_class.return_value = [1]
    data = {"f1": 0.0, "f2": 1.0}
    result = predict_logic(data, mock_predictor)
    assert result["proba"] == 0.2
    assert result["classe"] == 1

def test_predict_logic_missing_feature():
    "vérifie que "
    mock_predictor = Mock()
    mock_predictor.predict_proba.side_effect = KeyError("Feature manquante")
    data = {"f1": 0.0}
    with pytest.raises(KeyError):
        predict_logic(data, mock_predictor)

def test_predict_logic_unexpected_error():
    mock_predictor = Mock()
    mock_predictor.predict_proba.side_effect = RuntimeError("Erreur inattendue du modèle")
    data = {"f1": 0.0, "f2": 1.0}
    with pytest.raises(RuntimeError):
        predict_logic(data, mock_predictor)

# Test credit_decision
def test_credit_decision():
    assert credit_decision(1) == "refusé"
    assert credit_decision(0) == "accordé"






