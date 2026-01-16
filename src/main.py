from fastapi import FastAPI
import pandas as pd
from src.model_loader import predictor

app = FastAPI(title="Credit Scoring API")

# Charger les 5 clients dès le démarrage
clients_df = pd.read_csv("5_clients.csv", index_col=0)

@app.get("/")
def read_root():
    return {"message": "API Credit Scoring active"}

@app.get("/clients")
def get_clients():
    # Retourne les 5 clients sous forme de dictionnaire
    return clients_df.to_dict(orient="index")

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = df[predictor.top_features]  
        proba = predictor.predict_proba(df)[0]
        classe = predictor.predict_class(df)[0]
        return {"proba": float(proba), "classe": int(classe)}
    except Exception as e:
        return {"error": str(e)}

