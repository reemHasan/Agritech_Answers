"""
Crop Yield Prediction & Recommendation API
=============================================
Serves the trained Ridge model (registered in MLflow as
'crop_yield_ridge_model') via two endpoints:

  POST /predict   - yield prediction for a single chosen crop + parcel context
  POST /recommend - ranks all known crops by predicted yield, for a given
                    parcel context (crop not required as input)

The model is loaded once at startup from a local artifact directory
"""
from app.models import Crop, RecommendRequest, RecommendResponse, ParcelContext, PredictRequest, PredictResponse, CropRecommendation
from app.logger import configure_logging

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
load_dotenv()

configure_logging()
logger = logging.getLogger("crop_yield_api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#print(MODEL_PATH)
ALL_CROPS = [c.value for c in Crop]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_PATH = os.environ.get("MODEL_PATH", "model")
    if not os.path.isfile(MODEL_PATH):
        logger.error("model_not_found", extra={"model_path": MODEL_PATH})
        raise RuntimeError(
            f"Model file '{MODEL_PATH}' not found. Export the registered "
            f"MLflow model to this path before starting the API"
        )
    start = time.perf_counter()
    app.state.model = joblib.load(MODEL_PATH)
    app.state.model_name = "Ridge"
    logger.info("model_loaded", extra={
        "model_path": MODEL_PATH,
        "load_duration_ms": round((time.perf_counter() - start) * 1000, 1),
    })
    yield
    app.state.model = None
    logger.info("model_unloaded")


app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predicts crop yield and recommends the most profitable crop "
                "for a given parcel's growing conditions.",
    version="1.0.0",
    license_info={"name": "MIT",},
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    """Assigns a request ID (echoed back in the response header) and logs
    one line per request with method, path, status, and latency -- gives
    every endpoint basic observability for free, without repeating logging
    boilerplate in each handler."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_completed", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
    })
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def predict_yield(context: ParcelContext, crop: str) -> float:
    row = context_to_row(context, crop)
    prediction = app.state.model.predict(row)[0]
    # Yield cannot be negative; the model is linear and could in principle
    # extrapolate below zero for extreme/unusual input combinations.
    return max(0.0, float(prediction))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health_check():
    """Check if the API is running and if the model is loaded."""
    try:                      
        return {"status": "ok", "model_loaded": app.state.model is not None, "version": "1.0.0",
                "model name": app.state.model_name,
                "available_endpoints": {
                "/predict": "Predicts yield for one specific crop under the given parcel conditions.",
                "/recommend":"Simulates yield for every known crop under the given parcel conditions",
                "/docs": "/docs",},
                }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest, http_request: Request):
    """Predicts yield for one specific crop under the given parcel conditions."""
    if app.state.model is None:
        logger.error("predict_failed_model_not_loaded",
                      extra={"request_id": getattr(http_request.state, "request_id", None)})
        raise HTTPException(status_code=503, detail="Model not loaded")

    predicted_yield = predict_yield(request, request.Crop.value)

    logger.info("predict", extra={
        "request_id": getattr(http_request.state, "request_id", None),
        "crop": request.Crop.value,
        "region": request.Region.value,
        "predicted_yield_tons_per_hectare": round(predicted_yield, 3),
    })

    return PredictResponse(
        crop=request.Crop.value,
        predicted_yield_tons_per_hectare=round(predicted_yield, 3),
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["recommendation"])
def recommend(request: RecommendRequest, http_request: Request):
    """Simulates yield for every known crop under the given parcel conditions,
    and returns them ranked by predicted yield, descending."""
    if app.state.model is None:
        logger.error("recommend_failed_model_not_loaded",
                      extra={"request_id": getattr(http_request.state, "request_id", None)})
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for crop in ALL_CROPS:
        predicted_yield = predict_yield(request, crop)
        results.append({"crop": crop, "predicted_yield_tons_per_hectare": predicted_yield})

    results.sort(key=lambda r: r["predicted_yield_tons_per_hectare"], reverse=True)

    recommendations = [
        CropRecommendation(
            crop=r["crop"],
            predicted_yield_tons_per_hectare=round(r["predicted_yield_tons_per_hectare"], 3),
            rank=i + 1,
        )
        for i, r in enumerate(results)
    ]

    logger.info("recommend", extra={
        "request_id": getattr(http_request.state, "request_id", None),
        "region": request.Region.value,
        "top_crop": recommendations[0].crop,
        "top_yield_tons_per_hectare": recommendations[0].predicted_yield_tons_per_hectare,
    })

    return RecommendResponse(recommendations=recommendations)