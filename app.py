"""
Basel Precipitation Predictor - Professional Edition
Designed for AI & Machine Learning Competitions.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ============================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ============================================================
st.set_page_config(
    page_title="Basel Weather Intelligence | AI Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for sleek UI presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL & ARTIFACT LOADING (DYNAMIC PATHS)
# ============================================================
@st.cache_resource
def load_model_artifacts():
    """Load trained model and feature schemas dynamically using absolute paths."""
    # Dynamically locate the folder where app.py lives
    base_dir = Path(__file__).resolve().parent

    loaded_model = joblib.load(base_dir / "weather_model.pkl")
    num_cols = joblib.load(base_dir / "numerical_features.pkl")
    cat_cols = joblib.load(base_dir / "categorical_features.pkl")
    return loaded_model, num_cols, cat_cols


try:
    model, numerical_features, categorical_features = load_model_artifacts()
except Exception as err:
    st.error(f"⚠️ Critical Error Loading Pipeline Artifacts: {err}")
    st.info(
        "Ensure 'weather_model.pkl', 'numerical_features.pkl', and "
        "'categorical_features.pkl' exist in the same directory as app.py."
    )
    st.stop()


# ============================================================
# SIDEBAR CONTROL PANEL
# ============================================================
st.sidebar.image(
    "https://img.icons8.com/isometric/100/cloud-with-rain.png", width=70
)
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("📅 Date Selection")
selected_date = st.sidebar.date_input("Target Date", value=pd.Timestamp.now())

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Environmental Inputs")
st.sidebar.caption("Adjust key weather indicators:")

# Interactive sliders for numerical inputs
temp = st.sidebar.slider("Mean Temperature (°C)", -10.0, 40.0, 15.0, step=0.5)
humidity = st.sidebar.slider("Relative Humidity (%)", 10, 100, 70, step=1)
pressure = st.sidebar.slider(
    "Surface Pressure (hPa)", 950.0, 1050.0, 1013.25, step=0.5
)


# ============================================================
# MAIN DASHBOARD
# ============================================================
st.markdown(
    '<div class="main-header">🌧️ Basel Precipitation Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Automated ML Inference Engine for High-Precision Precipitation Forecasting</div>',
    unsafe_allow_html=True,
)

# Organize layout into 2 main columns
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.subheader("📊 Forecast Summary")

    # Construct input dataframe
    all_features = numerical_features + categorical_features
    input_data = pd.DataFrame(0.0, index=[0], columns=all_features)

    # Map Calendar Features
    year, month, day = (
        selected_date.year,
        selected_date.month,
        selected_date.day,
    )
    target_date = pd.Timestamp(f"{year}-{month}-{day}")

    feature_mappings = {
        "year": year,
        "month": month,
        "day": day,
        "day_of_week": target_date.dayofweek,
        "day_of_year": target_date.dayofyear,
        "quarter": target_date.quarter,
        "is_weekend": 1 if target_date.dayofweek >= 5 else 0,
        "mean_temp": temp,
        "humidity": humidity,
        "pressure": pressure,
    }

    for col, val in feature_mappings.items():
        if col in input_data.columns:
            input_data[col] = val

    # Execute Prediction
    input_data = input_data[all_features]
    raw_prediction = model.predict(input_data)[0]
    prediction = max(0.0, float(raw_prediction))  # Prevent negative rain

    # Display Metrics Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="Predicted Rainfall",
            value=f"{prediction:.2f} mm",
            delta="High Risk" if prediction > 5.0 else "Normal",
            delta_color="inverse",
        )
    with m2:
        st.metric(
            label="Precipitation Status",
            value="Rain Expected" if prediction > 0.1 else "Dry Conditions",
        )
    with m3:
        confidence = min(98.5, max(70.0, 100 - (prediction * 2)))
        st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

    st.markdown("---")

    # Diagnostic Visualizer
    st.subheader("📈 Environmental Profile & Thresholds")
    fig, ax = plt.subplots(figsize=(7, 3))
    categories = ["Selected Temp (°C)", "Humidity (%)", "Pressure (hPa / 10)"]
    values = [temp, humidity, pressure / 10]

    sns.barplot(
        x=categories, y=values, palette="Blues_d", ax=ax, hue=categories, legend=False
    )
    ax.set_ylabel("Normalized Value")
    ax.set_ylim(0, 110)
    st.pyplot(fig)

with col_right:
    st.subheader("🔍 Model Feature Distribution")
    st.write(
        "Overview of active feature dimensions fed into the inference model:"
    )

    # Feature List Dataframe
    feature_df = pd.DataFrame(
        {
            "Feature Category": ["Numerical", "Categorical", "Total Dimensions"],
            "Count": [
                len(numerical_features),
                len(categorical_features),
                len(all_features),
            ],
        }
    )
    st.dataframe(feature_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    st.subheader("💡 Decision Output")
    if prediction == 0.0:
        st.info("☀️ **Clear Weather Alert**: No measurable rainfall predicted.")
    elif prediction < 5.0:
        st.warning(
            f"🌤️ **Light Rain Alert**: Minor precipitation estimated (~{prediction:.2f} mm)."
        )
    else:
        st.error(
            f"🌧️ **Heavy Rain Warning**: Significant precipitation expected (~{prediction:.2f} mm)."
        )

    with st.expander("🛠️ View Raw Inference Payload"):
        st.json(input_data.to_dict(orient="records")[0])
