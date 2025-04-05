import pandas as pd
import joblib
import streamlit as st

# Load model and features
model_data = joblib.load('../models/house_price_model.pkl')
model = model_data['model']
expected_features = model_data['features']

# User input
user_input = {
    'OverallQual': 7,
    'GrLivArea': 1710,
    'GarageCars': 2,
    'TotalBsmtSF': 856,
    'Neighborhood': 'CollgCr'
}

# Transform input into DataFrame
X_input = pd.DataFrame([user_input])

# One-Hot Encoding for Neighborhood
X_input = pd.get_dummies(X_input)

# Add missing columns
for col in expected_features:
    if col not in X_input.columns:
        X_input[col] = 0

# Ensure correct column order
X_input = X_input[expected_features]

# Predict
prediction = model.predict(X_input)[0]

st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")
