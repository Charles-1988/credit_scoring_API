from fastapi import FastAPI
from pydantic import create_model
import pandas as pd
from pathlib import Path
from src.model_loader import ModelPredictor

BASE_DIR = Path(__file__).resolve().parent.parent

# Charger le modèle et les clients
predictor = ModelPredictor(
    BASE_DIR / "models/best_model_lightgbm.pkl",
    BASE_DIR / "data/top_features.csv",
    threshold=0.09
)
clients_df = pd.read_csv(BASE_DIR / "data/five_clients.csv", index_col=0)

# FastAPI
app = FastAPI(title="Credit Scoring API")

# Validation Pydantic
ClientData = create_model(
    "ClientData",
    **{feat: (float, ...) for feat in predictor.top_features}
)

# Logique métier séparée pour tests unitaires
def predict_logic(data: dict, predictor_instance):
    df = pd.DataFrame([data])
    proba = predictor_instance.predict_proba(df)[0]
    classe = predictor_instance.predict_class(df)[0]
    return {"proba": float(proba), "classe": int(classe)}

def credit_decision(classe: int):
    return "refusé" if classe == 1 else "accordé"

# Endpoints
@app.get("/")
def read_root():
    return {"message": "API Credit Scoring active"}

@app.get("/clients")
def get_clients():
    return clients_df.to_dict(orient="index")

@app.post("/predict")
def predict(data: ClientData):
    try:
        return predict_logic(data.dict(), predictor)
    except KeyError as e:
        return {"error": f"Feature manquante: {e}"}
    except Exception as e:
        return {"error": f"Erreur serveur: {e}"}





