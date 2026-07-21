"""
Crop Yield Prediction & Recommendation API
=============================================
Serves the trained Ridge model (registered in MLflow as
'crop_yield_ridge_model') via two endpoints:

  POST /predict   - yield prediction for a single chosen crop + parcel context
  POST /recommend - ranks all known crops by predicted yield, for a given
                    parcel context (crop not required as input)

The model is loaded once at startup from a local artifact directory
(exported from MLflow beforehand -- see README note at the bottom of this
file / the Dockerfile), so the container has no runtime dependency on a
live MLflow tracking server.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import List

import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured (JSON) logging
# ---------------------------------------------------------------------------
# One JSON object per log line, written to stdout -- the standard contract
# for containerized services, so a log aggregator (ELK, Loki, CloudWatch,
# etc.) can parse and index fields directly instead of grepping free text.

# Attribute names LogRecord already carries by default -- anything else on
# the record came from an `extra={...}` kwarg passed to a logging call, and
# gets folded into the JSON output as its own field (e.g. request_id,
# duration_ms, crop).
_RESERVED_LOG_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Emit every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_LOG_ATTRS}
        log_obj.update(extras)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


configure_logging()
logger = logging.getLogger("crop_yield_api")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "model")

# Fixed vocabularies, matching the categories the model was trained on
# (dataset1). Using Enums here gives free request validation AND populates
# FastAPI's auto-generated docs with the exact allowed values -- a bad
# category value is rejected with a clear 422 error before it ever reaches
# the model, instead of silently producing a meaningless prediction via
# OneHotEncoder's handle_unknown="ignore".

class Region(str, Enum):
    north = "North"
    south = "South"
    east = "East"
    west = "West"


class SoilType(str, Enum):
    sandy = "Sandy"
    clay = "Clay"
    loam = "Loam"
    silt = "Silt"
    peaty = "Peaty"
    chalky = "Chalky"


class WeatherCondition(str, Enum):
    sunny = "Sunny"
    rainy = "Rainy"
    cloudy = "Cloudy"


class Crop(str, Enum):
    wheat = "Wheat"
    rice = "Rice"
    maize = "Maize"
    barley = "Barley"
    soybean = "Soybean"
    cotton = "Cotton"


ALL_CROPS = [c.value for c in Crop]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ParcelContext(BaseModel):
    """Shared parcel conditions, used by both endpoints."""
    Region: Region
    Soil_Type: SoilType
    Rainfall_mm: float = Field(..., ge=0, description="Annual rainfall in mm")
    Temperature_Celsius: float = Field(..., description="Average temperature in Celsius")
    Fertilizer_Used: bool
    Irrigation_Used: bool
    Weather_Condition: WeatherCondition
    Days_to_Harvest: int = Field(..., gt=0, description="Days from planting to harvest")

    class Config:
        json_schema_extra = {
            "example": {
                "Region": "West",
                "Soil_Type": "Loam",
                "Rainfall_mm": 850.0,
                "Temperature_Celsius": 24.5,
                "Fertilizer_Used": True,
                "Irrigation_Used": True,
                "Weather_Condition": "Sunny",
                "Days_to_Harvest": 120,
            }
        }


class PredictRequest(ParcelContext):
    Crop: Crop


class PredictResponse(BaseModel):
    crop: str
    predicted_yield_tons_per_hectare: float


class RecommendRequest(ParcelContext):
    pass


class CropRecommendation(BaseModel):
    crop: str
    predicted_yield_tons_per_hectare: float
    rank: int


class RecommendResponse(BaseModel):
    recommendations: List[CropRecommendation]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

model = None  # populated at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.isdir(MODEL_PATH):
        logger.error("model_directory_not_found", extra={"model_path": MODEL_PATH})
        raise RuntimeError(
            f"Model directory '{MODEL_PATH}' not found. Export the registered "
            f"MLflow model to this path before starting the API (see Dockerfile)."
        )
    start = time.perf_counter()
    model = mlflow.sklearn.load_model(MODEL_PATH)
    logger.info("model_loaded", extra={
        "model_path": MODEL_PATH,
        "load_duration_ms": round((time.perf_counter() - start) * 1000, 1),
    })
    yield
    model = None
    logger.info("model_unloaded")


app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predicts crop yield and recommends the most profitable crop "
                "for a given parcel's growing conditions.",
    version="1.0.0",
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
    prediction = model.predict(row)[0]
    # Yield cannot be negative; the model is linear and could in principle
    # extrapolate below zero for extreme/unusual input combinations.
    return max(0.0, float(prediction))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest, http_request: Request):
    """Predicts yield for one specific crop under the given parcel conditions."""
    if model is None:
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
    if model is None:
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