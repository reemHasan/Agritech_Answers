"""
Hyperparameter Tuning — Ridge
================================
Only Ridge is tuned here. At baseline, Ridge and Linear Regression were
identical (regularization negligible at alpha=1.0), Ridge had essentially
zero overfit gap (0.00002 vs LightGBM's 0.004 and Random Forest's 0.08),
and CatBoost trailed Ridge by only 0.0002 R² without any tuning. Given
Ridge's interpretability advantage for a farmer-facing report (direct,
explainable coefficients) and its already-negligible overfit risk, further
search effort was concentrated on Ridge's alpha/solver rather than
committing to a full CatBoost search for a marginal, unconfirmed gain.

Each trial (one hyperparameter combination, evaluated via 5-fold CV) is
logged as its own MLflow run, so the search trajectory is fully visible
in the MLflow UI. A summary run logs the sorted results table and a
hyperparameter-sensitivity plot.

Requirements:
    pip install mlflow scikit-learn pandas numpy matplotlib scipy
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.catboost

from sklearn.model_selection import KFold, ParameterSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import loguniform, randint, uniform


# ---------------------------------------------------------------------------
# Config (kept consistent with train_model_comparison.py)
# ---------------------------------------------------------------------------

TRAIN_PATH = "train.csv"
TARGET_COL = "Yield_tons_per_hectare"

NUMERIC_FEATURES = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
BOOLEAN_FEATURES = ["Fertilizer_Used", "Irrigation_Used"]
CATEGORICAL_FEATURES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

N_SPLITS = 5
RANDOM_STATE = 42
N_ITER = 25  # number of random hyperparameter combinations tried per model

MLFLOW_EXPERIMENT_NAME = "crop_yield_hyperparameter_tuning"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sample(path: str, sample_size: int = 100_000) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_crops = df["Crop"].nunique()
    per_crop = sample_size // n_crops
    sample = (
        df.groupby("Crop", observed=True, group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_crop), random_state=RANDOM_STATE))
    )
    return sample.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing (same logic as train_model_comparison.py)
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("bool", "passthrough", BOOLEAN_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def get_param_distributions() -> dict:
    return {
        "ridge": {
            "alpha": loguniform(1e-3, 1e2),
            "solver": ["auto", "svd", "cholesky", "lsqr"],
        },
    }


def build_model(params: dict):
    return Ridge(random_state=RANDOM_STATE, **params)


# ---------------------------------------------------------------------------
# CV evaluation for one hyperparameter combination
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def cv_score_trial(params: dict, X: pd.DataFrame, y: pd.Series):
    preprocessor = build_preprocessor()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_r2, fold_mae = [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(params)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)

        metrics = evaluate(y_val, preds)
        fold_r2.append(metrics["r2"])
        fold_mae.append(metrics["mae"])

    return {
        "mean_val_r2": float(np.mean(fold_r2)),
        "std_val_r2": float(np.std(fold_r2)),
        "mean_val_mae": float(np.mean(fold_mae)),
    }


# ---------------------------------------------------------------------------
# Randomized search for one model, each trial logged as its own MLflow run
# ---------------------------------------------------------------------------

def run_random_search(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    param_distributions = get_param_distributions()["ridge"]
    sampler = ParameterSampler(
        param_distributions, n_iter=N_ITER, random_state=RANDOM_STATE
    )

    trial_records = []

    for trial_idx, params in enumerate(sampler):
        with mlflow.start_run(run_name=f"ridge_trial_{trial_idx}"):
            mlflow.set_tag("tuning_group", "ridge")
            mlflow.set_tag("trial_index", trial_idx)
            mlflow.log_params(params)

            result = cv_score_trial(params, X, y)
            mlflow.log_metrics(result)

            record = {"trial": trial_idx, **params, **result}
            trial_records.append(record)

            print(f"[ridge] trial {trial_idx}: params={params} "
                  f"-> val_r2={result['mean_val_r2']:.4f}")

    return pd.DataFrame(trial_records)


# ---------------------------------------------------------------------------
# Summary run: sorted trial table + sensitivity plots, per model
# ---------------------------------------------------------------------------

def log_tuning_summary(model_name: str, trials_df: pd.DataFrame, param_names: list):
    trials_df_sorted = trials_df.sort_values("mean_val_r2", ascending=False).reset_index(drop=True)

    csv_path = f"{model_name}_tuning_trials.csv"
    trials_df_sorted.to_csv(csv_path, index=False)

    html_path = f"{model_name}_tuning_trials.html"
    trials_df_sorted.to_html(html_path, index=False)

    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4.5))
    if n_params == 1:
        axes = [axes]

    for ax, param in zip(axes, param_names):
        is_numeric = pd.api.types.is_numeric_dtype(trials_df_sorted[param])
        if is_numeric:
            ax.scatter(trials_df_sorted[param], trials_df_sorted["mean_val_r2"], alpha=0.7)
            if trials_df_sorted[param].min() > 0 and trials_df_sorted[param].max() / max(trials_df_sorted[param].min(), 1e-9) > 20:
                ax.set_xscale("log")
        else:
            trials_df_sorted.boxplot(column="mean_val_r2", by=param, ax=ax)
            ax.set_title("")
        ax.set_xlabel(param)
        ax.set_ylabel("Validation R²")

    plt.suptitle(f"{model_name}: Hyperparameter Sensitivity ({N_ITER} random trials)")
    plt.tight_layout()
    chart_path = f"{model_name}_tuning_sensitivity.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    best_row = trials_df_sorted.iloc[0]

    with mlflow.start_run(run_name=f"{model_name}_tuning_summary"):
        mlflow.set_tag("tuning_group", model_name)
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(chart_path)
        mlflow.log_metric("best_val_r2", best_row["mean_val_r2"])
        mlflow.log_metric("best_val_mae", best_row["mean_val_mae"])
        mlflow.log_params({p: best_row[p] for p in param_names})
        mlflow.set_tag("best_params", json.dumps({p: str(best_row[p]) for p in param_names}))

    return {p: best_row[p] for p in param_names}, best_row["mean_val_r2"]


# ---------------------------------------------------------------------------
# Refit the overall best model on the full sample and log it as final
# ---------------------------------------------------------------------------

def log_final_model(best_params: dict, X: pd.DataFrame, y: pd.Series):
    preprocessor = build_preprocessor()
    model = build_model(best_params)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y)

    with mlflow.start_run(run_name="ridge_final_tuned"):
        mlflow.set_tag("tuning_group", "ridge")
        mlflow.set_tag("stage", "final_tuned_model")
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(pipeline.named_steps["model"], artifact_path="model")

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_sample(TRAIN_PATH, sample_size=100_000)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    param_distributions = get_param_distributions()

    print(f"\n=== Tuning ridge ({N_ITER} random trials) ===")
    trials_df = run_random_search(X, y)
    best_params, best_r2 = log_tuning_summary(
        "ridge", trials_df, list(param_distributions["ridge"].keys())
    )
    print(f"[ridge] best trial: R2={best_r2:.4f}, params={best_params}")

    log_final_model(best_params, X, y)

    with open("best_tuned_model.json", "w") as f:
        json.dump({
            "model": "ridge",
            "params": {k: str(v) for k, v in best_params.items()},
            "val_r2": best_r2,
        }, f, indent=2)


if __name__ == "__main__":
    main()