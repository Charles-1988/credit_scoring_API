
import joblib
import pandas as pd
from pathlib import Path

class ModelPredictor:
    def __init__(self, model_path=None, top_features_path=None, threshold=0.5):
        # base_path = dossier racine du projet
        base_path = Path(__file__).parent.parent  

        # Chemin du modèle
        if model_path is None:
            model_path = base_path / "models/best_model_lightgbm.pkl"

        # Chemin du fichier top features
        if top_features_path is None:
            top_features_path = base_path / "data/top_features.csv"

        # Charger le modèle et les features
        self.model = joblib.load(model_path)
        self.top_features = pd.read_csv(top_features_path)["feature"].tolist()
        self.threshold = threshold

    def predict_proba(self, X):
        X_sel = X[self.top_features]
        return self.model.predict_proba(X_sel)[:, 1]

    def predict_class(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


# Instanciation globale pour l’API
predictor = ModelPredictor()

