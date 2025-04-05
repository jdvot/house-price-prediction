# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------- Load model properly ---------
@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'house_price_model.pkl')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

model = load_model()

# List of neighborhoods (simplified version)
neighborhoods = [
    'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel',
    'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer'
]

# --------- Streamlit App Layout ---------
st.set_page_config(page_title="🏡 House Price Prediction", page_icon="🏡", layout="centered")

st.title("🏡 House Price Predictor")
st.markdown("### Predict the sale price of a house based on key features.")
st.write("---")

# --------- Input Features Section ---------
st.header("🔢 Input House Features")

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider('Overall Quality (1 = Poor, 10 = Excellent)', 1, 10, 5)
    GarageCars = st.slider('Garage Capacity (0-5 cars)', 0, 5, 2)
    Neighborhood = st.selectbox('Neighborhood', neighborhoods)

with col2:
    GrLivArea = st.number_input('Ground Living Area (sq ft)', 500, 5000, 1500)
    TotalBsmtSF = st.number_input('Total Basement Area (sq ft)', 0, 4000, 1000)

# --------- Prediction Button ---------
st.write("---")
if st.button("📈 Predict Sale Price"):

    # Prepare input
    input_dict = {
        'OverallQual': OverallQual,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF
    }
    
    # Add all neighborhoods as 0
    for n in neighborhoods:
        input_dict[f'Neighborhood_{n}'] = 0

    # Set selected neighborhood to 1
    if f'Neighborhood_{Neighborhood}' in input_dict:
        input_dict[f'Neighborhood_{Neighborhood}'] = 1

    input_data = pd.DataFrame([input_dict])

    # Display summary before prediction
    st.subheader("🔎 Your Inputs Recap:")
    st.dataframe(input_data)

    # Predict
    prediction = model.predict(input_data)
    predicted_price = int(prediction[0])

    st.success(f"💵 Estimated Sale Price: **${predicted_price:,}**")
    st.balloons()

# --------- Footer ---------
st.write("---")
st.caption("Created with ❤️ by Julien Devot · Powered by Random Forest Regressor")
st.caption("Baseline RMSE ≈ 29,000 $ · R² ≈ 0.89 🚀")
