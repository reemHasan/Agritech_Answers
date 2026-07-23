"""
Crop Yield Prediction & Recommendation — Streamlit UI
========================================================
Calls the FastAPI backend's /predict and /recommend endpoints.

Configure the backend URL via the API_URL environment variable (or
Streamlit secrets, see below) -- defaults to localhost for local dev
against `uvicorn main:app --reload` running in ../api.
"""

import os

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Resolution order: Streamlit secrets (recommended for deployed apps, set
# via the platform's secrets manager) -> environment variable -> local
# default. st.secrets raises if no secrets.toml exists at all, hence the
# try/except rather than a plain .get().
try:
    API_URL = st.secrets["API_URL"]
except (FileNotFoundError, KeyError):
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

REQUEST_TIMEOUT_SECONDS = 15

# Fallback UI options, matching main.py's Enums exactly. If
# compute_field_bounds.py has been run and ui_options.json is present
# alongside this file, those (real, data-derived) values are used instead.
DEFAULT_UI_OPTIONS = {
    "categorical": {
        "Region": ["East", "North", "South", "West"],
        "Soil_Type": ["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"],
        "Crop": ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"],
        "Weather_Condition": ["Cloudy", "Rainy", "Sunny"],
    },
    "numeric": {
        "Rainfall_mm": {"ui_slider_min": 0, "ui_slider_max": 1000, "ui_slider_default": 500},
        "Temperature_Celsius": {"ui_slider_min": 0, "ui_slider_max": 40, "ui_slider_default": 22},
        "Days_to_Harvest": {"ui_slider_min": 1, "ui_slider_max": 200, "ui_slider_default": 120},
    },
}


@st.cache_data
def load_ui_options() -> dict:
    import json
    options_path = os.path.join(os.path.dirname(__file__), "ui_options.json")
    if os.path.exists(options_path):
        with open(options_path) as f:
            return json.load(f)
    return DEFAULT_UI_OPTIONS


UI_OPTIONS = load_ui_options()


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_predict(payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_URL}. Is it running and reachable?")
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        st.error(f"API error: {detail}")
    return None


def call_recommend(payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/recommend", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_URL}. Is it running and reachable?")
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        st.error(f"API error: {detail}")
    return None


# ---------------------------------------------------------------------------
# Shared parcel-context input widgets
# ---------------------------------------------------------------------------

def parcel_context_inputs(key_prefix: str) -> dict:
    """Renders the shared parcel condition inputs and returns them as a
    dict matching the API's ParcelContext schema."""
    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Region", UI_OPTIONS["categorical"]["Region"], key=f"{key_prefix}_region")
        soil_type = st.selectbox("Soil Type", UI_OPTIONS["categorical"]["Soil_Type"], key=f"{key_prefix}_soil")
        weather = st.selectbox("Weather Condition", UI_OPTIONS["categorical"]["Weather_Condition"], key=f"{key_prefix}_weather")

    with col2:
        fertilizer = st.checkbox("Fertilizer Used", value=True, key=f"{key_prefix}_fertilizer")
        irrigation = st.checkbox("Irrigation Used", value=True, key=f"{key_prefix}_irrigation")

    rainfall_cfg = UI_OPTIONS["numeric"]["Rainfall_mm"]
    rainfall = st.slider(
        "Rainfall (mm/year)",
        min_value=float(rainfall_cfg["ui_slider_min"]),
        max_value=float(rainfall_cfg["ui_slider_max"]),
        value=float(rainfall_cfg["ui_slider_default"]),
        key=f"{key_prefix}_rainfall",
    )

    temp_cfg = UI_OPTIONS["numeric"]["Temperature_Celsius"]
    temperature = st.slider(
        "Average Temperature (°C)",
        min_value=float(temp_cfg["ui_slider_min"]),
        max_value=float(temp_cfg["ui_slider_max"]),
        value=float(temp_cfg["ui_slider_default"]),
        key=f"{key_prefix}_temp",
    )

    days_cfg = UI_OPTIONS["numeric"]["Days_to_Harvest"]
    days_to_harvest = st.slider(
        "Days to Harvest",
        min_value=int(days_cfg["ui_slider_min"]),
        max_value=int(days_cfg["ui_slider_max"]),
        value=int(days_cfg["ui_slider_default"]),
        key=f"{key_prefix}_days",
    )

    return {
        "Region": region,
        "Soil_Type": soil_type,
        "Rainfall_mm": rainfall,
        "Temperature_Celsius": temperature,
        "Fertilizer_Used": fertilizer,
        "Irrigation_Used": irrigation,
        "Weather_Condition": weather,
        "Days_to_Harvest": days_to_harvest,
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Crop Yield Advisor", page_icon="🌾", layout="centered")
st.title("🌾 Crop Yield Advisor")
st.caption("Estimate yield for a specific crop, or find the most profitable crop for your parcel.")

tab_predict, tab_recommend = st.tabs(["📈 Predict Yield", "🏆 Recommend Best Crop"])

with tab_predict:
    st.subheader("Predict yield for a specific crop")
    crop = st.selectbox("Crop", UI_OPTIONS["categorical"]["Crop"], key="predict_crop")
    context = parcel_context_inputs(key_prefix="predict")

    if st.button("Predict Yield", type="primary", key="predict_button"):
        payload = {**context, "Crop": crop}
        with st.spinner("Contacting the model..."):
            result = call_predict(payload)
        if result:
            st.metric(
                label=f"Predicted yield — {result['crop']}",
                value=f"{result['predicted_yield_tons_per_hectare']:.2f} t/ha",
            )

with tab_recommend:
    st.subheader("Find the most profitable crop for your parcel")
    st.caption("No crop needed here — every known crop is simulated and ranked for you.")
    context = parcel_context_inputs(key_prefix="recommend")

    if st.button("Recommend Best Crop", type="primary", key="recommend_button"):
        with st.spinner("Simulating all crops..."):
            result = call_recommend(context)
        if result:
            recommendations = result["recommendations"]
            df = pd.DataFrame(recommendations)

            best = recommendations[0]
            st.success(
                f"🏆 Best choice: **{best['crop']}** "
                f"— {best['predicted_yield_tons_per_hectare']:.2f} t/ha"
            )

            st.bar_chart(
                df.set_index("crop")["predicted_yield_tons_per_hectare"],
                color="#2ca02c",
            )

            st.dataframe(
                df.rename(columns={
                    "crop": "Crop",
                    "predicted_yield_tons_per_hectare": "Predicted Yield (t/ha)",
                    "rank": "Rank",
                }),
                hide_index=True,
                use_container_width=True,
            )

st.divider()
st.caption(f"Connected to API: `{API_URL}`")
