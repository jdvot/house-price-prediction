import streamlit as st
import joblib
import os
import pandas as pd

# -- Prepare input function
def prepare_input(user_input: dict, expected_features: list) -> pd.DataFrame:
    df = pd.DataFrame([user_input])

    if 'Neighborhood' in df.columns:
        neighborhood_encoded = pd.get_dummies(df['Neighborhood'], prefix='Neighborhood')
        df = pd.concat([df.drop('Neighborhood', axis=1), neighborhood_encoded], axis=1)

    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0

    df = df[expected_features]

    return df

# -- Load model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, '..', 'models', 'house_price_model.pkl')
model_data = joblib.load(model_path)

model = model_data['model']
expected_features = model_data['features']

# -- Streamlit Interface
st.title("🏡 House Price Prediction")

user_input = {
    'OverallQual': st.slider('Overall Quality', 1, 10, 5),
    'GrLivArea': st.number_input('Ground Living Area (sq ft)', min_value=100, max_value=5000, value=1500),
    'GarageCars': st.slider('Garage Cars', 0, 4, 2),
    'TotalBsmtSF': st.number_input('Total Basement (sq ft)', min_value=0, max_value=3000, value=800),
    'Neighborhood': st.selectbox('Neighborhood', [
        'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel',
        'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer'
    ])
}

# -- Checkbox pour afficher debug
show_debug = st.checkbox('Afficher les valeurs transformées (debug)')

# -- PREDICTION
if st.button('Predict'):
    x_input = prepare_input(user_input, expected_features)

    if show_debug:
        st.subheader("🔍 Valeurs après préparation (x_input)")
        st.write(x_input)

    prediction = model.predict(x_input)[0]
    st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")
