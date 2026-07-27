"""
Model loading and prediction helpers for the Crop Yield API.
"""
import os

import joblib
import pandas as pd

from api.src.logger import logger
from api.src.pydantic_models import ParcelContext
 
 
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

CATEGORICAL_FEATURES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]


def _parent_feature(col_name: str) -> str:
    """Maps a post-ColumnTransformer column name back to its original
    feature (e.g. 'cat__Soil_Type_Clay' -> 'Soil_Type'), so one-hot
    dummies collapse back into one human-readable feature per category."""
    if col_name.startswith("cat__"):
        stripped = col_name[len("cat__"):]
        for cat in CATEGORICAL_FEATURES:
            if stripped.startswith(cat + "_"):
                return cat
        return stripped
    if col_name.startswith("num__"):
        return col_name[len("num__"):]
    if col_name.startswith("bool__"):
        return col_name[len("bool__"):]
    return col_name


def compute_feature_contributions(model, row: pd.DataFrame) -> dict:
    """Computes exact, closed-form per-feature contributions for a single
    prediction -- equivalent to SHAP values for a LINEAR model
    (contribution_i = coefficient_i * transformed_feature_i), with no
    `shap` dependency needed. This is exact, not an approximation, because
    the model is linear: base_value + sum(contributions) always equals
    the model's raw prediction exactly.

    Only works if `model` is a full sklearn Pipeline with named steps
    "preprocessor" and "model" (a linear model exposing .coef_ /
    .intercept_) -- e.g. the real Ridge pipeline this project trains and
    deploys. Returns None if the model doesn't match that shape (e.g. a
    test double), so callers can treat missing explanations gracefully
    rather than erroring.

    Note on baseline: numeric features are centered by StandardScaler
    during preprocessing, so their contribution's implicit baseline is
    exactly the training-set mean (matching SHAP's own definition
    precisely for those features). One-hot/boolean features use an
    implicit baseline of 0 ("category absent" / "False"), the standard
    convention for linear-model coefficient attribution -- a
    simplification relative to a full SHAP background-average baseline
    for those specific features, though the sum-to-prediction identity
    holds regardless of this framing choice.
    """
    try:
        preprocessor = model.named_steps["preprocessor"]
        linear_model = model.named_steps["model"]
        coefficients = linear_model.coef_
        intercept = linear_model.intercept_
    except (AttributeError, KeyError, TypeError):
        return None

    transformed = preprocessor.transform(row)[0]
    feature_names = preprocessor.get_feature_names_out()

    grouped: dict = {}
    for name, coef, value in zip(feature_names, coefficients, transformed):
        parent = _parent_feature(name)
        grouped[parent] = grouped.get(parent, 0.0) + float(coef) * float(value)

    base_value = float(intercept)
    return {
        "base_value": base_value,
        "contributions": grouped,
        "prediction": base_value + sum(grouped.values()),
    }  