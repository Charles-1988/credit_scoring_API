import streamlit as st
import pandas as pd
import requests

# URL de ton API
API_URL = "https://ton-api-render.onrender.com/predict"  
API_CLIENTS_URL = "https://ton-api-render.onrender.com/clients"  

st.title("Credit Scoring - 5 clients")

# Récupérer les clients depuis l'API
try:
    response = requests.get(API_CLIENTS_URL)
    response.raise_for_status()
    clients_df = pd.DataFrame(response.json())
except Exception as e:
    st.error(f"Impossible de récupérer les clients depuis l'API: {e}")
    clients_df = pd.DataFrame()  

if not clients_df.empty:
    # Sélectionner un client
    client_id = st.selectbox("Sélectionnez un client", clients_df.index)

    # Afficher les features du client
    client_data = clients_df.loc[client_id]
    st.subheader("Features du client")
    edited_data = {}
    for col in clients_df.columns:
        edited_data[col] = st.number_input(col, value=float(client_data[col]))

    # Bouton pour prédire
    if st.button("Prédire"):
        try:
            pred_response = requests.post(API_URL, json=edited_data)
            pred_response.raise_for_status()
            prediction = pred_response.json()
            st.success(f"Prédiction du risque de crédit: {prediction['prediction']}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")



