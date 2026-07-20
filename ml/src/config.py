
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = BASE_DIR/"data/processed_data/train.csv"
VAL_PATH = BASE_DIR/"data/processed_data/val.csv"
TEST_PATH = BASE_DIR/"data/processed_data/test.csv"
TARGET_COL = "Yield_tons_per_hectare"

NUMERIC_FEATURES = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
BOOLEAN_FEATURES = ["Fertilizer_Used", "Irrigation_Used"]
CATEGORICAL_FEATURES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
ENRICHED_FEATURES = ["ref_rainfall_mm_per_year", "ref_pesticides_tonnes", "ref_avg_temp"]
# ref_yield_tons_per_ha intentionally excluded from both feature sets: it is
# a yield-derived proxy and would constitute leakage if used as a predictor.
ml_artifact_folder = Path(__file__).resolve().parent.parent/"ml_artifact"
ml_artifact_folder.mkdir(exist_ok=True)
SAMPLE_SIZE = 100_000       # sample of train, for CV speed
N_SPLITS = 5
RANDOM_STATE = 42