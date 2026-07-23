# Fixed vocabularies, matching the categories the model was trained on
# (dataset1). Using Enums here gives free request validation AND populates
# FastAPI's auto-generated docs with the exact allowed values -- a bad
# category value is rejected with a clear 422 error before it ever reaches
# the model, instead of silently producing a meaningless prediction via
# OneHotEncoder's handle_unknown="ignore".

from enum import Enum
from typing import List
from pydantic import BaseModel, Field, StrictBool

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
    """Shared parcel conditions, used by both endpoints.
    """
    Region: Region
    Soil_Type: SoilType
    Rainfall_mm: float = Field(..., ge=100.00089622522204, le=999.998098221668, description="Annual rainfall in mm")
    Temperature_Celsius: float = Field(..., ge=15.000034141430271, le=39.99999662316004, description="Average temperature in Celsius")
    Fertilizer_Used: StrictBool
    Irrigation_Used: StrictBool
    Weather_Condition: WeatherCondition
    Days_to_Harvest: int = Field(..., ge=60, le=149, description="Days from planting to harvest")

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
                "Crop": "Wheat",
            }
        }


class PredictResponse(BaseModel):
    crop: str
    predicted_yield_tons_per_hectare: float = Field(..., ge=0)


class RecommendRequest(ParcelContext):
    pass


class CropRecommendation(BaseModel):
    crop: str
    predicted_yield_tons_per_hectare: float = Field(..., ge=0)
    rank: int


class RecommendResponse(BaseModel):
    recommendations: List[CropRecommendation]

