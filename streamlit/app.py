import streamlit as st
import os
import joblib
import pandas as pd
st.title("🏠 House Price Predictor 🚀")


# Get the absolute path to the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'house_price_model.pkl')

# Load model
model = joblib.load(model_path)
# Entrées utilisateur
OverallQual = st.slider('Overall Quality (1-10)', 1, 10, 5)
GrLivArea = st.number_input('Ground Living Area (sq ft)', 500, 5000, 1500)
GarageCars = st.slider('Garage Capacity (cars)', 0, 5, 2)
TotalBsmtSF = st.number_input('Total Basement Area (sq ft)', 0, 4000, 1000)

# Prédiction
input_data = pd.DataFrame([{
    'OverallQual': OverallQual,
    'GrLivArea': GrLivArea,
    'GarageCars': GarageCars,
    'TotalBsmtSF': TotalBsmtSF
}])

prediction = model.predict(input_data)

st.write(f"💵 Predicted Sale Price: ${int(prediction[0]):,}")
