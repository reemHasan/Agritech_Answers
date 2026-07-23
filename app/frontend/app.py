"""
Crop Yield Prediction & Recommendation — Streamlit UI
========================================================
Calls the FastAPI backend's /predict and /recommend endpoints.

Configure the backend URL via the API_URL environment variable (or
Streamlit secrets) -- defaults to localhost for local dev
against `uvicorn main:app --reload` running in app/backend/api.
"""

import os
import pandas as pd
import requests
import streamlit as st
import altair as alt

st.set_page_config(page_title="Crop Yield Advisor", page_icon="🌾", layout="centered")
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

try:
    API_URL = st.secrets["API_URL"]
except (FileNotFoundError, KeyError):
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

# render.yaml links API_URL via fromService's "host" property, which
# returns a bare hostname (e.g. "crop-yield-api.onrender.com"), not a full
# URL -- add the scheme if it's missing, so this works whether API_URL
# came from Render's auto-linking, a manually-set env var|streamlit.secrets with a full URL,
# or the localhost default above.
if API_URL and not API_URL.startswith(("http://", "https://")):
    API_URL = f"https://{API_URL}"

REQUEST_TIMEOUT_SECONDS = 15


@st.cache_data
def load_ui_options() -> dict:
    import json
    options_path = os.path.join(os.path.dirname(__file__), "ui_options.json")
    print("i'm here")
    if os.path.exists(options_path):
        with open(options_path) as f:
            return json.load(f)
    else:
        raise ImportError("File not found, please check that ui_options.json is extracted by runing the script ml/src/utils_app")

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
    """Renders the shared parcel condition inputs, in order: dropdowns,
    then checkboxes, then sliders. Returns them as a dict matching the
    API's ParcelContext schema."""
 
    # --- Dropdowns ---------------------------------------------------
    region = st.selectbox("Region", UI_OPTIONS["categorical"]["Region"], key=f"{key_prefix}_region")
    soil_type = st.selectbox("Soil Type", UI_OPTIONS["categorical"]["Soil_Type"], key=f"{key_prefix}_soil")
    weather = st.selectbox("Weather Condition", UI_OPTIONS["categorical"]["Weather_Condition"], key=f"{key_prefix}_weather")
 
    # --- Checkboxes ----------------------------------------------------
    col,col1, col2, col3 = st.columns([1, 1, 1, 1])
    with col1:
        fertilizer = st.checkbox("Fertilizer Used", value=True, key=f"{key_prefix}_fertilizer")
    with col2:
        irrigation = st.checkbox("Irrigation Used", value=True, key=f"{key_prefix}_irrigation")
 
    # --- Sliders ---------------------------------------------------------
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
st.title("🌾 Crop Yield Advisor")
st.caption("Estimate yield for a specific crop, or find the most profitable crop for your parcel.")
 
# Streamlit's default tabs render small and tightly packed. This targets
# the underlying Base Web components (data-baseweb attributes are
# Streamlit's own internal markup, stable across recent versions but not
# a guaranteed public API -- if a future Streamlit upgrade changes its
# component library, this CSS may need updating).
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        padding: 0 8px;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
tab_predict, tab_recommend = st.tabs(["📈 Predict Yield", "🏆 Recommend Best Crop"])
 
with tab_predict:
    st.subheader("Predict yield for a specific crop")
    crop = st.selectbox("Crop", UI_OPTIONS["categorical"]["Crop"], key="predict_crop")
    context = parcel_context_inputs(key_prefix="predict")
 
    # Centered button: an empty side column on each side narrows the
    # middle column, so the button (which stretches to its container's
    # width) ends up visually centered instead of spanning full width.
    left, center, right = st.columns([1, 2, 1])
    with center:
        predict_clicked = st.button("Predict Yield", type="primary", key="predict_button", use_container_width=True)
 
    if predict_clicked:
        payload = {**context, "Crop": crop}
        with st.spinner("Contacting the model..."):
            result = call_predict(payload)
        if result:
            left, center, right = st.columns([1, 2, 1])
            with center:
                st.markdown(
                    f"""
                    <p style="font-size:20px; background:#e1e1e1">
                        <strong>Predicted yield for {result['crop']} is </strong>
                        {result['predicted_yield_tons_per_hectare']:.2f} t/ha
                    </p>
                    """,
                    unsafe_allow_html=True,
                )
 
with tab_recommend:
    st.subheader("Find the most profitable crop for your parcel")
    st.caption("No crop needed here — every known crop is simulated and ranked for you.")
    context = parcel_context_inputs(key_prefix="recommend")
 
    left, center, right = st.columns([1, 2, 1])
    with center:
        recommend_clicked = st.button("Recommend Best Crop", type="primary", key="recommend_button", use_container_width=True)
 
    if recommend_clicked:
        with st.spinner("Simulating all crops..."):
            result = call_recommend(context)
        if result:
            recommendations = result["recommendations"]
            df = pd.DataFrame(recommendations)
 
            best = recommendations[0]
            left, center, right = st.columns([1, 3, 1])
            with center:
                st.success(
                    f" 🏆 Best choice: **{best['crop']}** "
                    f"with predicted yield **{best['predicted_yield_tons_per_hectare']:.3f}** t/ha"
                )
            st.write("")
            ranked_table, ranked_chart = st.columns([2, 2])
            with ranked_table:
                st.dataframe(
                    df.rename(columns={
                        "crop": "Crop",
                        "predicted_yield_tons_per_hectare": "Predicted Yield (t/ha)",
                        "rank": "Rank",
                    }),
                    hide_index=True,
                    use_container_width=True,
                )
            with ranked_chart:
                df["Color"] = df["rank"].apply(
                    lambda r: "Best crop" if r == 1 else "Other crops"
                )
                chart = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(
                        x=alt.X("crop:N", title="Crop", sort="-y"),
                        y=alt.Y("predicted_yield_tons_per_hectare:Q", title="Predicted Yield (t/ha)"),
                        color=alt.Color(
                            "Color:N",
                            scale=alt.Scale(
                                domain=["Best crop", "Other crops"],
                                range=["#2ca02c", "#bdbdbd"]  # green and gray
                            ),
                            legend=None,
                        ),
                        tooltip=[
                            "crop",
                            "predicted_yield_tons_per_hectare",
                            "rank",
                        ],
                    )
                )

                st.altair_chart(chart, use_container_width=True)

st.divider()
st.caption(f"Connected to API: `{API_URL}`")