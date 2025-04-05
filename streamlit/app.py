import os
import joblib
import pandas as pd
import streamlit as st

# Charger modèle
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, '..', 'models', 'house_price_model.pkl')

model_data = joblib.load(model_path)
model = model_data['model']
expected_features = model_data['features']

# Input utilisateur
user_input = {
    'OverallQual': 7,
    'GrLivArea': 1710,
    'GarageCars': 2,
    'TotalBsmtSF': 856,
    'Neighborhood': 'CollgCr'
}

X_input = pd.DataFrame([user_input])

# One-Hot Encoding
if 'Neighborhood' in X_input.columns:
    X_input = pd.get_dummies(X_input)

# Ajouter colonnes manquantes
for col in expected_features:
    if col not in X_input.columns:
        X_input[col] = 0

# Réordonner les colonnes
X_input = X_input[expected_features]

# Prédire
prediction = model.predict(X_input)[0]

st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")
