import joblib
import pandas as pd
from pathlib import Path

class ModelPredictor:
    def __init__(self, model_path=None, top_features_path=None, threshold=0.09):
        base_path = Path(__file__).parent.parent
        self.model = joblib.load(model_path or base_path / "models/best_model_lightgbm.pkl")
        self.top_features = pd.read_csv(top_features_path or base_path / "data/top_features.csv")["feature"].tolist()
        self.threshold = threshold

    def predict_proba(self, X: pd.DataFrame):
        missing = set(self.top_features) - set(X.columns)
        if missing:
            raise KeyError(f"Feature(s) manquante(s) : {missing}")
        return self.model.predict_proba(X[self.top_features])[:, 1]

    def predict_class(self, X: pd.DataFrame):
        return (self.predict_proba(X) >= self.threshold).astype(int)




