"""
Crop Yield Prediction & Recommendation API
=============================================
Serves the trained Ridge model via two endpoints:

  POST /predict   - yield prediction for a single chosen crop + parcel context
  POST /recommend - ranks all known crops by predicted yield, for a given
                    parcel context (crop not required as input)

The model is loaded once at startup from a local artifact directory
See pydantic_models.py for request/response schemas, logger.py for
structured logging setup, and helpers.py for model loading and prediction
logic.
"""
from api.logger import logger
from api.helpers import  predict_yield, load_model
from api.pydantic_models import (
    ALL_CROPS,
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    CropRecommendation,
)
import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_PATH = os.environ.get("MODEL_PATH", "model")
    print("MODEL_PATH",MODEL_PATH)
    start = time.perf_counter()
    app.state.model = load_model(MODEL_PATH)
    #app.state.model = joblib.load(MODEL_PATH)
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
# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# allow_credentials=False is required here: browsers reject credentialed
# requests (cookies/auth headers) to a wildcard "*" origin per the CORS
# spec, so allow_origins=["*"] + allow_credentials=True is a combination
# that silently doesn't work as intended. This API doesn't use cookies or
# browser-stored auth, so disabling credentials is the correct fix rather
# than restricting origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def health_check(response: Response):
    """Check if the API is running and if the model is loaded."""
    try:
        if  app.state.model is not None:
            return {"status": "ok", "model_loaded": True, "version": "1.0.0",
                "model name": app.state.model_name,
                "available_endpoints": {
                "/predict": "Predicts yield for one specific crop under the given parcel conditions.",
                "/recommend":"Simulates yield for every known crop under the given parcel conditions",
                "/docs": "/docs",},
                }
        else:
            response.status_code = 503
            return {"status": "unavailable", "model_loaded": False}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(request: PredictRequest, http_request: Request):
    """Predicts yield for one specific crop under the given parcel conditions."""
    if app.state.model is None:
        logger.error("predict_failed_model_not_loaded",
                      extra={"request_id": getattr(http_request.state, "request_id", None)})
        raise HTTPException(status_code=503, detail="Model not loaded")

    predicted_yield = predict_yield(model=app.state.model, context=request, crop=request.Crop.value)

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
        predicted_yield = predict_yield(model=app.state.model, context=request, crop=crop)
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