import streamlit as st
import pandas as pd
import requests

API_URL = "https://credit-scoring-api-tqja.onrender.com/predict"
API_CLIENTS_URL = "https://credit-scoring-api-tqja.onrender.com/clients"

st.title("Credit Scoring")

# Charger les clients
try:
    clients_df = pd.DataFrame.from_dict(requests.get(API_CLIENTS_URL).json(), orient="index")
except Exception as e:
    st.error(f"Erreur lors de la récupération des clients : {e}")
    st.stop()

# Sélection du client
client_id = st.selectbox("Choisir un client :", clients_df.index.tolist())
client_data = clients_df.loc[client_id]

# Colonnes pour bouton et résultat
col1, col2 = st.columns(2)

# Préparer le formulaire
inputs = {col: st.number_input(col, value=float(client_data[col])) for col in clients_df.columns}

# Bouton de prédiction
with col1:
    if st.button("Prédire"):
        try:
            res = requests.post(API_URL, json=inputs).json()
            if "error" in res:
                col2.error(f"Erreur API : {res['error']}")
            elif res["classe"] == 1:
                col2.error("Crédit refusé")
            else:
                col2.success("Crédit accordé")
        except Exception as e:
            col2.error(f"Erreur lors de l'appel API : {e}")











