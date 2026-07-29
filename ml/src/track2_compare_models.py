"""
Reversed Merge — Real-World Data (dataset2) as Primary
==========================================================
Inverts the original merge direction: dataset2 (real, FAO-style records)
is treated as the primary table, enriched with crop-level aggregates from
dataset1 (rainfall, temperature, and fertilizer/irrigation usage rates).
The same 5-model comparison used on dataset1 (Ridge, Linear Regression,
LightGBM, CatBoost, Random Forest) is run here via 5-fold CV, to see
whether the "linear relationship, low crop/enrichment importance" story
found on the synthetic dataset also holds on real-world data.

Key differences from the original direction, expected upfront:
  - Dataset2 is far smaller (thousands of rows, not 1M) -> results will
    have wider confidence intervals; interpret with that in mind.
  - dataset1-derived features become a CROP-LEVEL CONSTANT repeated across
    many dataset2 rows (same caveat as the original ablation, just mirrored).
  - Area (country) is a much richer categorical signal than dataset1's
    generic Region, so tree-based / CatBoost models may behave differently
    here than they did on the synthetic data.
  - Several dataset2 crops (Cassava, Plantains and others, Potatoes, Sweet
    potatoes, Yams) have NO dataset1 counterpart -> their enrichment
    columns will be NaN, handled via median imputation for Ridge/LightGBM/
    RandomForest, and natively by CatBoost.

"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.catboost
from mlflow import MlflowClient
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_COL = "yield_tons_per_ha"  # converted from hg/ha_yield for consistency

NATIVE_NUMERIC = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
NATIVE_CATEGORICAL = ["Area", "Item"]

ENRICHED_NUMERIC = [
    "d1_ref_rainfall_mm", "d1_ref_temperature_c",
    "d1_ref_fertilizer_rate", "d1_ref_irrigation_rate", "d1_ref_days_to_harvest",
]

ALL_NUMERIC = NATIVE_NUMERIC + ENRICHED_NUMERIC
ALL_CATEGORICAL = NATIVE_CATEGORICAL

N_SPLITS = 5
RANDOM_STATE = 42
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "reverse_merge_real_world_model_comparison"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    #return df
    # Only rows with a genuine dataset1 match, so both ablation arms are
    # compared on identical rows (mirrors the original ablation's approach).
    matched = df.dropna(subset=ENRICHED_NUMERIC).reset_index(drop=True)
    print(f"Matched rows (Wheat/Rice/Maize/Soybeans): {len(matched)} of {len(df)} total")
    return matched

# ---------------------------------------------------------------------------
# Preprocessing — per model, mirroring the original comparison script
# ---------------------------------------------------------------------------

def build_preprocessor(model_name: str) -> ColumnTransformer:
    """
    Numeric pipeline includes median imputation (crops with no dataset1
    match have NaN in the d1_ref_* columns) before scaling.
    CatBoost: categoricals passed through raw (native handling); others
    one-hot encoded. Area has ~100+ levels -- one-hot is still workable
    given dataset2's larger-than-one-row-per-country row count, but this
    is exactly the kind of high-cardinality case where CatBoost's native
    encoding is expected to have a real advantage this time.
    """
    numeric_transformer = Pipeline(steps=[
        #("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    if model_name == "catboost":
        categorical_transformer = Pipeline(steps=[
            ("to_str", FunctionTransformer(lambda X: X.astype(str))),
        ])
    else:
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, ALL_NUMERIC),
        ("cat", categorical_transformer, ALL_CATEGORICAL),
    ])


def catboost_categorical_indices() -> list:
    return list(range(len(ALL_NUMERIC), len(ALL_NUMERIC) + len(ALL_CATEGORICAL)))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def get_models() -> dict:
    return {
        "ridge": Ridge(alpha=1.0, solver="auto", max_iter=1000, random_state=RANDOM_STATE),
        "linear_regression": LinearRegression(n_jobs=-1),
        "lightgbm": LGBMRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8
        ),
        "catboost": CatBoostRegressor(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            random_state=RANDOM_STATE,
            cat_features=catboost_categorical_indices(),
            verbose=False,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1)
    }


# ---------------------------------------------------------------------------
# CV + MLflow logging (same pattern as train_model_comparison.py)
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def log_model_artifact(model_name: str, model):
    if model_name == "lightgbm":
        mlflow.lightgbm.log_model(model, artifact_path="model")
    elif model_name == "catboost":
        mlflow.catboost.log_model(model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(model, artifact_path="model")


def run_cv_experiment(model_name: str, model, X: pd.DataFrame, y: pd.Series, output_path:str):
    preprocessor = build_preprocessor(model_name)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    last_fitted_model = None

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("n_rows", len(X))
        mlflow.log_param("direction", "dataset2_primary_enriched_with_dataset1")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)

            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_metrics = evaluate(y_train, train_preds)
            val_metrics = evaluate(y_val, val_preds)

            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v, step=fold_idx)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=fold_idx)

            fold_rows.append({
                "fold": fold_idx,
                "train_r2": train_metrics["r2"], "val_r2": val_metrics["r2"],
                "train_mae": train_metrics["mae"], "val_mae": val_metrics["mae"],
            })
            last_fitted_model = pipeline.named_steps["model"]

            print(f"[{model_name}] fold {fold_idx}: "
                  f"train R2={train_metrics['r2']:.4f} | val R2={val_metrics['r2']:.4f}")

        fold_df = pd.DataFrame(fold_rows)
        fold_df["overfit_gap_r2"] = fold_df["train_r2"] - fold_df["val_r2"]
        html_path = output_path/f"{model_name}_fold_metrics.html"
        fold_df.to_html(html_path, index=False)
        mlflow.log_artifact(html_path)

        summary = {}
        for prefix in ["train", "val"]:
            for metric in ["r2", "mae"]:
                col = f"{prefix}_{metric}"
                summary[f"mean_{col}"] = float(fold_df[col].mean())
                summary[f"std_{col}"] = float(fold_df[col].std())
        summary["mean_overfit_gap_r2"] = float(fold_df["overfit_gap_r2"].mean())
        mlflow.log_metrics(summary)

        log_model_artifact(model_name, last_fitted_model)

        print(f"[{model_name}] mean val R2={summary['mean_val_r2']:.4f} "
              f"(+/- {summary['std_val_r2']:.4f}) | overfit gap={summary['mean_overfit_gap_r2']:.4f}\n")

        return summary


# ---------------------------------------------------------------------------
# Comparison chart
# ---------------------------------------------------------------------------

def log_model_comparison(comparison_df: pd.DataFrame, output_path:str):
    csv_path = output_path/"reverse_model_comparison_results.csv"
    comparison_df.to_csv(csv_path)

    models = comparison_df.index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(models, comparison_df["mean_val_r2"], yerr=comparison_df["std_val_r2"],
                capsize=5, color=colors)
    axes[0].set_title("Validation R² (± std across folds)")

    axes[1].bar(models, comparison_df["mean_val_mae"], color=colors)
    axes[1].set_title("Validation MAE")

    axes[2].bar(models, comparison_df["mean_overfit_gap_r2"], color=colors)
    axes[2].set_title("Overfit Gap (train R² − val R²)")
    axes[2].axhline(0, color="black", linewidth=0.8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    plt.suptitle("Reversed Merge — Model Comparison on Real-World Data (dataset2 primary)")
    plt.tight_layout()
    chart_path = output_path/"reverse_model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    with mlflow.start_run(run_name="comparison_summary"):
        mlflow.set_tag("comparison_group", "reverse_merge_5_models")
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(chart_path)
        for model_name in models:
            mlflow.log_metric(f"{model_name}_val_r2", comparison_df.loc[model_name, "mean_val_r2"])
        mlflow.set_tag("best_model", comparison_df["mean_val_r2"].idxmax())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = Path(__file__).resolve().parent.parent
    ml_artifact_folder = base_dir/"ml_artifact"/"reverse_merge"
    ml_artifact_folder.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    merged = load_data(base_dir/"../data/processed_data/reversed_crop_dataset.csv")

    X = merged[ALL_NUMERIC + ALL_CATEGORICAL].copy()
    y = merged[TARGET_COL].copy()

    results = {}
    for model_name, model in get_models().items():
        results[model_name] = run_cv_experiment(model_name, model, X, y, output_path=ml_artifact_folder)

    comparison_df = pd.DataFrame(results).T.sort_values("mean_val_r2", ascending=False)
    print("\n=== Reversed merge — model comparison (sorted by mean validation R2) ===")
    print(comparison_df[["mean_val_r2", "std_val_r2", "mean_val_mae", "mean_overfit_gap_r2"]])

    log_model_comparison(comparison_df,output_path=ml_artifact_folder)


if __name__ == "__main__":
    main()