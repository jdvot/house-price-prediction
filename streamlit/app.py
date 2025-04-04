import streamlit as st
import pandas as pd
import joblib
import os

# --------- Load model properly ---------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'house_price_model.pkl')
model = joblib.load(model_path)

# --------- Streamlit App Layout ---------
st.set_page_config(page_title="🏡 House Price Prediction", page_icon="🏡", layout="centered")

st.title("🏡 House Price Predictor")
st.markdown("### Predict the sale price of a house based on key features")
st.write("---")

# --------- Input Features Section ---------
st.header("🔢 Input Features")

col1, col2 = st.columns(2)

with col1:
    OverallQual = st.slider('Overall Quality (1 = Poor, 10 = Excellent)', 1, 10, 5)
    GarageCars = st.slider('Garage Capacity (0-5 cars)', 0, 5, 2)

with col2:
    GrLivArea = st.number_input('Ground Living Area (sq ft)', 500, 5000, 1500)
    TotalBsmtSF = st.number_input('Total Basement Area (sq ft)', 0, 4000, 1000)

# --------- Prediction Button ---------
st.write("---")
if st.button("📈 Predict Sale Price"):
    input_data = pd.DataFrame([{
        'OverallQual': OverallQual,
        'GrLivArea': GrLivArea,
        'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF
    }])

    prediction = model.predict(input_data)
    predicted_price = int(prediction[0])

    st.success(f"💵 Estimated Sale Price: **${predicted_price:,}**")
    st.balloons()

# --------- Footer ---------
st.write("---")
st.caption("Created with ❤️ by Julien Devot")
