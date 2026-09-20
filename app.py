"""Streamlit app for predicting precipitation in Basel."""

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# STREAMLIT INTERFACE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Weather Prediction App", layout="centered")
st.title("🌦️ Basel Precipitation Predictor")
st.write("Enter a target date to predict the expected precipitation levels.")


@st.cache_resource
def load_model_artifacts():
    """Load model artifacts safely from disk."""
    loaded_model = joblib.load("weather_model.pkl")
    num_cols = joblib.load("numerical_features.pkl")
    cat_cols = joblib.load("categorical_features.pkl")
    return loaded_model, num_cols, cat_cols


try:
    model, numerical_features, categorical_features = load_model_artifacts()
except (FileNotFoundError, KeyError, ValueError) as load_err:
    st.error(f"Could not load model files. Error details: {load_err}")
    st.info(
        "Make sure 'weather_model.pkl', 'numerical_features.pkl', and "
        "'categorical_features.pkl' are in the app folder."
    )
    st.stop()

# User Input Interface
st.subheader("Select Date for Prediction")
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2026)
with col2:
    month = st.slider("Month", min_value=1, max_value=12, value=6)
with col3:
    day = st.slider("Day", min_value=1, max_value=31, value=15)

if st.button("Predict Precipitation", type="primary"):
    all_features = numerical_features + categorical_features
    input_data = pd.DataFrame(0.0, index=[0], columns=all_features)

    if "year" in input_data.columns:
        input_data["year"] = year
    if "month" in input_data.columns:
        input_data["month"] = month
    if "day" in input_data.columns:
        input_data["day"] = day

    try:
        target_date = pd.Timestamp(f"{year}-{month}-{day}")
        if "day_of_week" in input_data.columns:
            input_data["day_of_week"] = target_date.dayofweek
        if "day_of_year" in input_data.columns:
            input_data["day_of_year"] = target_date.dayofyear
        if "quarter" in input_data.columns:
            input_data["quarter"] = target_date.quarter
        if "is_weekend" in input_data.columns:
            input_data["is_weekend"] = 1 if target_date.dayofweek >= 5 else 0
    except (ValueError, TypeError) as date_err:
        st.error(f"Invalid calendar date selected: {date_err}")
        st.stop()

    input_data = input_data[all_features]
    prediction = model.predict(input_data)[0]
    st.success(f"### Predicted Precipitation: **{prediction:.2f} mm**")
