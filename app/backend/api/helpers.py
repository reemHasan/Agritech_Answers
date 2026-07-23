"""
Model loading and prediction helpers for the Crop Yield API.
"""
import pandas as pd
from api.pydantic_models import ParcelContext
import joblib
import os 
from api.logger import logger
 
 
def load_model(model_path: str):
    """Loads the trained pipeline from a joblib file. A plain joblib export is
    the expected production path (lighter image, no MLflow runtime
    dependency needed just to load and serve the model)
    """
    if os.path.isfile(model_path):
        model = joblib.load(model_path)
        logger.info("model_loaded", extra={"load_method": "joblib"})
        return model
 
    logger.error("model_path_not_found", extra={"model_path": model_path})
    raise RuntimeError(
        f"'{model_path}' is not a file. Set MODEL_PATH to a valid joblib file "
        f"(e.g. ridge_pipeline.joblib), exported from a trained/registered "
        f"MLflow model (run ml/src/utils_app.py to export trained model)."
    )
def context_to_row(context: ParcelContext, crop: str) -> pd.DataFrame:
    """Builds a single-row DataFrame matching the exact column names and
    order the model's preprocessing pipeline was fitted on."""
    return pd.DataFrame([{
        "Rainfall_mm": context.Rainfall_mm,
        "Temperature_Celsius": context.Temperature_Celsius,
        "Days_to_Harvest": context.Days_to_Harvest,
        "Fertilizer_Used": context.Fertilizer_Used,
        "Irrigation_Used": context.Irrigation_Used,
        "Region": context.Region.value,
        "Soil_Type": context.Soil_Type.value,
        "Crop": crop,
        "Weather_Condition": context.Weather_Condition.value,
    }])


def predict_yield(model, context: ParcelContext, crop: str) -> float:
    row = context_to_row(context, crop)
    prediction = model.predict(row)[0]
    # Yield cannot be negative; the model is linear and could in principle
    # extrapolate below zero for extreme/unusual input combinations.
    return max(0.0, float(prediction))

  