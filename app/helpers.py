"""
Compute API Field Bounds & Streamlit UI Options from train.csv
==================================================================
Generates two things from the actual training data:

1. `ui_options.json` -- categorical dropdown values, and BOTH the true
   min/max (hard API validation bounds) and the 1st/99th percentile range
   (recommended Streamlit slider bounds -- tighter, so users aren't offered
   rare extreme values the model saw very little of) for each numeric
   feature.
2. Console output with ready-to-paste `Field(ge=..., le=...)` lines for
   main.py's ParcelContext, so the API's validation bounds are computed
   from real data instead of the placeholder values currently in main.py.

"""

import json
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
TRAIN_PATH = base_dir/"data/processed_data/train.csv"

NUMERIC_FEATURES = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
CATEGORICAL_FEATURES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]


def extract_limits():
    df = pd.read_csv(TRAIN_PATH)

    # --- Categorical dropdown options ------------------------------------
    categorical_options = {
        col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES
    }

    # --- Numeric bounds: true min/max (API) + 1st/99th percentile (UI) ---
    numeric_ranges = {}
    for col in NUMERIC_FEATURES:
        true_min = df[col].min()
        true_max = df[col].max()
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        median = df[col].median()

        is_int = col == "Days_to_Harvest"  # only integer field among these
        numeric_ranges[col] = {
            "api_min": int(true_min) if is_int else float(true_min),
            "api_max": int(true_max) if is_int else float(true_max),
            "ui_slider_min": int(p01) if is_int else round(float(p01), 1),
            "ui_slider_max": int(p99) if is_int else round(float(p99), 1),
            "ui_slider_default": int(median) if is_int else round(float(median), 1),
        }

    ui_config = {"categorical": categorical_options, "numeric": numeric_ranges}

    with open("./app/ui_options.json", "w") as f:
        json.dump(ui_config, f, indent=2)
    print("Saved ui_options.json\n")

    # --- Ready-to-paste Field(...) lines for main.py's ParcelContext -----
    print("=== Paste into main.py's ParcelContext (API validation, true min/max) ===\n")
    for col in NUMERIC_FEATURES:
        r = numeric_ranges[col]
        if col == "Days_to_Harvest":
            print(f'    Days_to_Harvest: int = Field(..., ge={r["api_min"]}, le={r["api_max"]}, '
                  f'description="Days from planting to harvest")')
        else:
            print(f'    {col}: float = Field(..., ge={r["api_min"]}, le={r["api_max"]})')

    print("\n=== Streamlit slider bounds (1st/99th percentile, tighter than API limits) ===\n")
    for col in NUMERIC_FEATURES:
        r = numeric_ranges[col]
        print(f'st.slider("{col}", min_value={r["ui_slider_min"]}, '
              f'max_value={r["ui_slider_max"]}, value={r["ui_slider_default"]})')

    print("\n=== Categorical options ===\n")
    for col, values in categorical_options.items():
        print(f"{col}: {values}")


if __name__ == "__main__":
    extract_limits()