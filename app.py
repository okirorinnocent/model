import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Weather Prediction App", layout="centered")
st.title("🌦️ Basel Precipitation Predictor")
st.write("Enter a target date to predict the expected precipitation levels.")

# 1. Load the saved model artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('weather_model.pkl')
    num_cols = joblib.load('numerical_features.pkl')
    cat_cols = joblib.load('categorical_features.pkl')
    return model, num_cols, cat_cols

try:
    model, numerical_features, categorical_features = load_model_artifacts()
except Exception as e:
    st.error("Could not load model files. Make sure they are in the same folder as app.py.")
    st.stop()

# 2. Create User Input Interface
st.subheader("Select Date for Prediction")
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2026)
with col2:
    month = st.slider("Month", min_value=1, max_value=12, value=6)
with col3:
    day = st.slider("Day", min_value=1, max_value=31, value=15)

# 3. Generate feature baseline safely
if st.button("Predict Precipitation", type="primary"):
    # Create an empty template matching your exact trained features
    all_features = numerical_features + categorical_features
    input_data = pd.DataFrame(0.0, index=[0], columns=all_features)
    
    # Inject the user's date selections
    if 'year' in input_data.columns: input_data['year'] = year
    if 'month' in input_data.columns: input_data['month'] = month
    if 'day' in input_data.columns: input_data['day'] = day
    if 'day_of_week' in input_data.columns: input_data['day_of_week'] = pd.Timestamp(f"{year}-{month}-{day}").dayofweek
    if 'day_of_year' in input_data.columns: input_data['day_of_year'] = pd.Timestamp(f"{year}-{month}-{day}").dayofyear
    if 'quarter' in input_data.columns: input_data['quarter'] = pd.Timestamp(f"{year}-{month}-{day}").quarter
    if 'is_weekend' in input_data.columns: input_data['is_weekend'] = 1 if pd.Timestamp(f"{year}-{month}-{day}").dayofweek >= 5 else 0

    # Ensure columns match training order exactly
    input_data = input_data[all_features]

    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Display the result nicely
    st.success(f"### Predicted Precipitation: **{prediction:.2f} mm**")
