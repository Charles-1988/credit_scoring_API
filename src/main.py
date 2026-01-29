from fastapi import FastAPI
from pydantic import create_model
import pandas as pd
from pathlib import Path
from src.model_loader import ModelPredictor

BASE_DIR = Path(__file__).resolve().parent.parent

predictor = ModelPredictor(
    BASE_DIR / "models/best_model_lightgbm.pkl",
    BASE_DIR / "data/top_features.csv",
    threshold=0.09
)

clients_df = pd.read_csv(BASE_DIR / "data/five_clients.csv", index_col=0)

app = FastAPI(title="Credit Scoring API")

# Schéma Pydantic
ClientData = create_model(
    "ClientData",
    **{feat: (float, ...) for feat in predictor.top_features}
)

@app.get("/")
def root():
    return {"status": "API Credit Scoring active"}

@app.get("/clients")
def get_clients():
    return clients_df.to_dict(orient="index")

@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.dict()])
    proba = predictor.predict_proba(df)[0]
    classe = predictor.predict_class(df)[0]

    return {
        "proba": float(proba),
        "classe": int(classe),
        "decision": "refusé" if classe == 1 else "accordé"
    }

@app.post("/explain")
def explain(data: ClientData):
    df = pd.DataFrame([data.dict()])
    shap_values = predictor.get_shap_values(df)[0]

    return {
        feat: float(val)
        for feat, val in zip(predictor.top_features, shap_values)
    }







