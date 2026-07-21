"""
Pytest suite for the Crop Yield API.

The real Ridge model is replaced with a small deterministic FakeModel before
main.py is imported, so these tests exercise routing, request validation,
ranking logic, and edge-case handling -- WITHOUT requiring a real trained
model artifact on disk. This makes the suite fast, fully deterministic, and
safe to run in CI without any model-training step beforehand.

Run with:
    pytest test_main.py -v
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# --- Environment + model patch must happen BEFORE `import main` -----------
# main.py reads MODEL_PATH from the environment at import time, and its
# lifespan handler checks os.path.isdir(MODEL_PATH) before loading -- so a
# real (even if empty) directory must exist at that path.
_TMP_MODEL_DIR = tempfile.mkdtemp()
os.environ["MODEL_PATH"] = _TMP_MODEL_DIR


class FakeModel:
    """Deterministic stand-in for the real Ridge pipeline. Returns a fixed
    yield per crop, ignoring all other input -- enough to test the API's
    routing, validation, and /recommend ranking logic without depending on
    real model weights."""

    CROP_BASE_YIELD = {
        "Wheat": 5.0,
        "Rice": 7.0,
        "Maize": 3.0,
        "Barley": 6.0,
        "Soybean": 4.0,
        "Cotton": 2.0,
    }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        crop = X["Crop"].iloc[0]
        return np.array([self.CROP_BASE_YIELD[crop]])


class NegativeFakeModel:
    """Always predicts a negative yield, to test the API's clamping logic."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([-3.5])


# Patch the loader function directly on the mlflow.sklearn module. main.py
# calls `mlflow.sklearn.load_model(...)` (fully qualified) rather than
# `from mlflow.sklearn import load_model`, so this attribute swap is picked
# up correctly regardless of import order, and persists for the whole test
# session without needing a context-managed mock around every test.
import mlflow.sklearn as _mlflow_sklearn  # noqa: E402
_mlflow_sklearn.load_model = lambda path: FakeModel()

from fastapi.testclient import TestClient  # noqa: E402
import main as main_module  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


VALID_CONTEXT = {
    "Region": "West",
    "Soil_Type": "Loam",
    "Rainfall_mm": 850.0,
    "Temperature_Celsius": 24.5,
    "Fertilizer_Used": True,
    "Irrigation_Used": True,
    "Weather_Condition": "Sunny",
    "Days_to_Harvest": 120,
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    def test_valid_request_returns_expected_yield(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Rice"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["crop"] == "Rice"
        assert body["predicted_yield_tons_per_hectare"] == 7.0

    def test_response_matches_different_crop(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Cotton"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["predicted_yield_tons_per_hectare"] == 2.0

    def test_invalid_crop_rejected(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Unicorn"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_region_rejected(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Wheat", "Region": "Atlantis"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_required_field_rejected(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Wheat"}
        del payload["Rainfall_mm"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_rainfall_rejected(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Wheat", "Rainfall_mm": -10}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_non_positive_days_to_harvest_rejected(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Wheat", "Days_to_Harvest": 0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_wrong_type_for_boolean_field_rejected(self, client):
        payload = {**VALID_CONTEXT, "Crop": "Wheat", "Fertilizer_Used": "yes"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predicted_yield_never_negative(self, monkeypatch, client):
        monkeypatch.setattr(main_module, "model", NegativeFakeModel())
        payload = {**VALID_CONTEXT, "Crop": "Wheat"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["predicted_yield_tons_per_hectare"] == 0.0

    def test_returns_503_when_model_not_loaded(self, monkeypatch, client):
        monkeypatch.setattr(main_module, "model", None)
        payload = {**VALID_CONTEXT, "Crop": "Wheat"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------

class TestRecommendEndpoint:
    def test_returns_all_known_crops(self, client):
        response = client.post("/recommend", json=VALID_CONTEXT)
        assert response.status_code == 200
        recommendations = response.json()["recommendations"]
        assert len(recommendations) == len(FakeModel.CROP_BASE_YIELD)
        returned_crops = {r["crop"] for r in recommendations}
        assert returned_crops == set(FakeModel.CROP_BASE_YIELD.keys())

    def test_sorted_descending_by_yield(self, client):
        response = client.post("/recommend", json=VALID_CONTEXT)
        yields = [r["predicted_yield_tons_per_hectare"]
                  for r in response.json()["recommendations"]]
        assert yields == sorted(yields, reverse=True)

    def test_ranks_are_sequential_starting_at_1(self, client):
        response = client.post("/recommend", json=VALID_CONTEXT)
        ranks = [r["rank"] for r in response.json()["recommendations"]]
        assert ranks == list(range(1, len(FakeModel.CROP_BASE_YIELD) + 1))

    def test_top_recommendation_is_highest_yield_crop(self, client):
        response = client.post("/recommend", json=VALID_CONTEXT)
        top = response.json()["recommendations"][0]
        expected_best_crop = max(FakeModel.CROP_BASE_YIELD, key=FakeModel.CROP_BASE_YIELD.get)
        assert top["crop"] == expected_best_crop
        assert top["rank"] == 1

    def test_crop_field_not_required_in_request(self, client):
        # /recommend must work WITHOUT a Crop field -- it's the whole point
        # of the endpoint (simulate all crops, don't require one as input).
        assert "Crop" not in VALID_CONTEXT
        response = client.post("/recommend", json=VALID_CONTEXT)
        assert response.status_code == 200

    def test_missing_required_field_rejected(self, client):
        payload = dict(VALID_CONTEXT)
        del payload["Soil_Type"]
        response = client.post("/recommend", json=payload)
        assert response.status_code == 422

    def test_returns_503_when_model_not_loaded(self, monkeypatch, client):
        monkeypatch.setattr(main_module, "model", None)
        response = client.post("/recommend", json=VALID_CONTEXT)
        assert response.status_code == 503