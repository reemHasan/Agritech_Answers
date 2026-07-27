"""
Tests for helpers.compute_feature_contributions -- validates the actual
linear algebra against a small, real, fitted pipeline,
since this is the one piece of logic worth checking for numerical
correctness rather than just API routing behavior.
"""

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

from api.src.helpers import compute_feature_contributions


def _make_tiny_pipeline():
    """A small but real Ridge pipeline, structurally identical to the
    production one (same step names, same column-name conventions), fitted
    on a handful of synthetic rows -- enough to validate the math without
    needing the real trained model artifact."""
    numeric_features = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
    boolean_features = ["Fertilizer_Used", "Irrigation_Used"]
    categorical_features = ["Region", "Soil_Type", "Crop", "Weather_Condition"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("bool", "passthrough", boolean_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0)),
    ])

    train_df = pd.DataFrame({
        "Rainfall_mm": [200, 500, 800, 950, 300],
        "Temperature_Celsius": [15, 20, 25, 30, 18],
        "Days_to_Harvest": [90, 110, 130, 150, 100],
        "Fertilizer_Used": [True, False, True, True, False],
        "Irrigation_Used": [True, True, False, True, False],
        "Region": ["West", "East", "North", "South", "West"],
        "Soil_Type": ["Loam", "Clay", "Sandy", "Silt", "Loam"],
        "Crop": ["Wheat", "Rice", "Maize", "Barley", "Wheat"],
        "Weather_Condition": ["Sunny", "Rainy", "Cloudy", "Sunny", "Rainy"],
    })
    y_train = pd.Series([3.2, 4.5, 5.1, 6.8, 3.9])

    pipeline.fit(train_df, y_train)
    return pipeline


def _sample_row():
    return pd.DataFrame([{
        "Rainfall_mm": 850.0,
        "Temperature_Celsius": 24.5,
        "Days_to_Harvest": 120,
        "Fertilizer_Used": True,
        "Irrigation_Used": True,
        "Region": "West",
        "Soil_Type": "Loam",
        "Crop": "Wheat",
        "Weather_Condition": "Sunny",
    }])


class TestComputeFeatureContributions:
    def test_contributions_sum_to_prediction(self):
        """The core identity that makes this a valid linear-model
        explanation: base_value + sum(contributions) must exactly equal
        the model's own raw prediction for that row."""
        pipeline = _make_tiny_pipeline()
        row = _sample_row()

        result = compute_feature_contributions(pipeline, row)
        actual_prediction = pipeline.predict(row)[0]

        reconstructed = result["base_value"] + sum(result["contributions"].values())
        assert reconstructed == pytest.approx(actual_prediction, abs=1e-9)
        assert result["prediction"] == pytest.approx(actual_prediction, abs=1e-9)

    def test_onehot_contributions_grouped_to_parent_feature(self):
        """One-hot dummies (e.g. cat__Crop_Wheat) should collapse back to
        their parent feature name (Crop), not appear as separate,
        fragmented entries."""
        pipeline = _make_tiny_pipeline()
        row = _sample_row()

        result = compute_feature_contributions(pipeline, row)

        for parent in ["Region", "Soil_Type", "Crop", "Weather_Condition"]:
            assert parent in result["contributions"]
        # No leftover raw one-hot column names should appear
        assert not any(k.startswith("cat__") for k in result["contributions"])

    def test_numeric_and_boolean_features_present(self):
        pipeline = _make_tiny_pipeline()
        row = _sample_row()

        result = compute_feature_contributions(pipeline, row)

        for feature in ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest",
                         "Fertilizer_Used", "Irrigation_Used"]:
            assert feature in result["contributions"]

    def test_returns_none_for_non_pipeline_model(self):
        """A model that isn't a full sklearn Pipeline (e.g. a lightweight
        test double used elsewhere in the test suite) should fail
        gracefully -- return None -- rather than raising."""
        class NotAPipeline:
            def predict(self, X):
                return [1.0]

        result = compute_feature_contributions(NotAPipeline(), _sample_row())
        assert result is None