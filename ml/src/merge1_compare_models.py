"""
Model comparison script — Crop Yield Prediction
=================================================
Compares Ridge Regression, LightGBM, and CatBoost on the consolidated,
preprocessed training set, using 5-fold cross-validation. Each model gets
its own best-suited preprocessing (CatBoost uses native categorical
handling; the others use one-hot encoding). Train vs. validation metrics
are logged for every fold to trace overfitting, and a per-fold summary
table is logged to MLflow as an artifact.

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from src.config import TRAIN_PATH, TARGET_COL, NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES, ENRICHED_FEATURES, SAMPLE_SIZE, N_SPLITS, RANDOM_STATE

# Kept out of the model on purpose: validated via 5-fold CV + paired t-test
# to add no measurable predictive value at parcel-level granularity, and
# would introduce an unnecessary runtime dependency on dataset2 for the API.
EXCLUDED_FEATURES = [
    "ref_rainfall_mm_per_year",
    "ref_pesticides_tonnes",
    "ref_avg_temp",
    "ref_yield_tons_per_ha",
]

ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "crop_yield_model_comparison"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df

def load_sample(path: str, sample_size: int = 100_000) -> pd.DataFrame:
    """Stratified sample by Crop, so all 6 crops (including Barley/Cotton)
    stay proportionally represented at reduced size — useful for fast
    iteration before committing to a full-train run."""
    df = load_data(path)
    n_crops = df["Crop"].nunique()
    per_crop = sample_size // n_crops

    sample = (
        df.groupby("Crop", observed=True, group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), per_crop), random_state=RANDOM_STATE))
    )
    print("Crop distribution in the sample:\n")
    print(sample["Crop"].value_counts(normalize=True))
    return sample.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing — model-specific, so each model gets its best-suited setup
# ---------------------------------------------------------------------------

def build_preprocessor(model_name: str) -> ColumnTransformer:
    """
    - Ridge / LightGBM: numeric features scaled, categoricals one-hot encoded.
      Ridge needs numeric input and benefits from scaling (regularization is
      scale-sensitive). LightGBM works fine with one-hot too, kept consistent
      with Ridge here for simplicity.
    - CatBoost: numeric features scaled (harmless, no-op for a tree model,
      kept only for consistency/logging uniformity), categoricals passed
      through RAW (not one-hot encoded) — CatBoost's native categorical
      handling (ordered target statistics) generally outperforms one-hot,
      especially for higher-cardinality columns, and avoids the dimensionality
      blow-up one-hot causes. This is what "best possible version of each
      model" means in practice for CatBoost specifically.

      CatBoost's cat_features implementation requires the raw categorical
      columns to be string dtype, not pandas 'category' or mixed dtype.
      That cast is scoped to CatBoost's own transformer branch here (via a
      small FunctionTransformer), rather than applied to the shared X
      upfront in main() — Ridge/LightGBM never see or need this cast.
    """
    from sklearn.preprocessing import FunctionTransformer

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    if model_name == "catboost":
        numeric_transformer = "passthrough"
        categorical_transformer = Pipeline(steps=[
            ("to_str", FunctionTransformer(lambda X: X.astype(str))),
        ])
    elif model_name in ["lightgbm","randomforest"]:
        numeric_transformer = "passthrough"
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
    else:
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor



# CatBoost needs to know which columns (by position, post-ColumnTransformer)
# are categorical, since they're passed through raw as strings.
def catboost_categorical_indices() -> list:
    n_before_cat = len(NUMERIC_FEATURES) + len(BOOLEAN_FEATURES)
    return list(range(n_before_cat, n_before_cat + len(CATEGORICAL_FEATURES)))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Baseline (lightly-set, not yet tuned) hyperparameters — the winner gets
    a full hyperparameter search in the next step.

    Note: plain OLS `LinearRegression` has no `max_iter`/`solver` params
    (it's a closed-form least-squares solve). `Ridge` is used instead so
    those parameters are meaningful — it also regularizes, which is a fairer
    "best possible" linear baseline than unregularized OLS.
    """
    return {
        "linearRegression": LinearRegression(
            n_jobs=-1
        ),
        "ridge": Ridge(
            alpha=1.0,
            solver="auto",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
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
        "randomforest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1)
    }


# ---------------------------------------------------------------------------
# CV training + MLflow logging
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


def log_fold_table(model_name: str, fold_rows: list, artifact_path:str):
    """Logs the per-fold train-vs-val metrics table as an MLflow artifact,
    both as an HTML table (easy to screenshot/open) and as a native
    MLflow table (mlflow.log_table) when available."""
    fold_df = pd.DataFrame(fold_rows)
    fold_df["overfit_gap_r2"] = fold_df["train_r2"] - fold_df["val_r2"]

    html_path = artifact_path/f"{model_name}_fold_metrics.html"
    fold_df.to_html(html_path, index=False)
    mlflow.log_artifact(html_path)

    try:
        mlflow.log_table(data=fold_df, artifact_file=f"{model_name}_fold_metrics.json")
    except AttributeError:
        # Older MLflow versions without log_table support
        csv_path = artifact_path/f"{model_name}_fold_metrics.csv"
        fold_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

    return fold_df


def run_cv_experiment(model_name: str, model, X: pd.DataFrame, y: pd.Series, artifact:str):
    # step1: build processor
    preprocessor = build_preprocessor(model_name)
    # step2: start cv with loging each fold info
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []
    last_fitted_model = None

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_rows", len(X))
        mlflow.log_params({f"hp_{k}": v for k, v in model.get_params().items()
                            if k != "cat_features"})  # skip long list param

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ])
            pipeline.fit(X_train, y_train)

            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_metrics = evaluate(y_train, train_preds)
            val_metrics = evaluate(y_val, val_preds)

            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v, step=fold_idx)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=fold_idx)
            # get eval metrics for each fold 
            fold_rows.append({
                "fold": fold_idx,
                "train_r2": train_metrics["r2"], "val_r2": val_metrics["r2"],
                "train_mae": train_metrics["mae"], "val_mae": val_metrics["mae"],
                "train_rmse": train_metrics["rmse"], "val_rmse": val_metrics["rmse"],
            })

            last_fitted_model = pipeline.named_steps["model"]
            print(f"[{model_name}] fold {fold_idx}: "
                  f"train R2={train_metrics['r2']:.4f} | val R2={val_metrics['r2']:.4f} "
                  f"| gap={train_metrics['r2'] - val_metrics['r2']:.4f}")

        fold_df = log_fold_table(model_name, fold_rows, artifact_path=artifact)

        summary = {}
        for prefix in ["train", "val"]:
            for metric in ["r2", "mae", "rmse"]:
                col = f"{prefix}_{metric}"
                summary[f"mean_{col}"] = float(fold_df[col].mean())
                summary[f"std_{col}"] = float(fold_df[col].std())
        summary["mean_overfit_gap_r2"] = float(fold_df["overfit_gap_r2"].mean())
        mlflow.log_metrics(summary)

        log_model_artifact(model_name, last_fitted_model)

        print(f"[{model_name}] mean val R2={summary['mean_val_r2']:.4f} "
              f"(+/- {summary['std_val_r2']:.4f}) | "
              f"overfit gap={summary['mean_overfit_gap_r2']:.4f}\n")

        return summary

# ---------------------------------------------------------------------------
# Cross-model comparison: plot + CSV, logged as its own MLflow run
# ---------------------------------------------------------------------------

def log_model_comparison(comparison_df: pd.DataFrame, atrifact_path:str):
    """Builds a 3-panel comparison figure (val R2 with error bars, val MAE,
    overfit gap) plus the full metrics table, and logs both to a dedicated
    'comparison_summary' MLflow run"""

    csv_path = atrifact_path/"model_comparison_results.csv"
    comparison_df.to_csv(csv_path)

    html_path = atrifact_path/"model_comparison_results.html"
    comparison_df.to_html(html_path)

    models = comparison_df.index.tolist()
    colors = ["#5578B1", "#423D7E", "#DD8452", "#55A868", "#C92680" ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(models, comparison_df["mean_val_r2"],
                yerr=comparison_df["std_val_r2"], capsize=5, color=colors)
    axes[0].set_title("Validation R² (± std across folds)")
    axes[0].set_ylabel("R²")
    ymin = comparison_df["mean_val_r2"].min() - 0.01
    ymax = comparison_df["mean_val_r2"].max() + 0.005
    axes[0].set_ylim(ymin, ymax)

    axes[1].bar(models, comparison_df["mean_val_mae"], color=colors)
    axes[1].set_title("Validation MAE")
    axes[1].set_ylabel("MAE (tons/hectare)")

    axes[2].bar(models, comparison_df["mean_overfit_gap_r2"], color=colors)
    axes[2].set_title("Overfit Gap (train R² − val R²)")
    axes[2].set_ylabel("R² gap")
    axes[2].axhline(0, color="black", linewidth=0.8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Model Comparison — LinearReg vs Ridge vs LightGBM vs CatBoost vs RandomForest")
    plt.tight_layout()
    chart_path = atrifact_path/"model_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
"""
    with mlflow.start_run(run_name="comparison_summary"):
        mlflow.set_tag("comparison_group", "ridge_lightgbm_catboost")
        mlflow.log_artifact(csv_path)
        mlflow.log_artifact(html_path)
        mlflow.log_artifact(chart_path)

        for model_name in models:
            mlflow.log_metric(f"{model_name}_val_r2", comparison_df.loc[model_name, "mean_val_r2"])
            mlflow.log_metric(f"{model_name}_val_mae", comparison_df.loc[model_name, "mean_val_mae"])
            mlflow.log_metric(f"{model_name}_overfit_gap", comparison_df.loc[model_name, "mean_overfit_gap_r2"])

        best_model_name = comparison_df["mean_val_r2"].idxmax()
        mlflow.set_tag("best_model", best_model_name)"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    ml_aritfact_path = BASE_DIR/"ml_artifact"
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    # Use a stratified 100K sample for faster iteration; switch to
    # load_data(TRAIN_PATH) for the final run on the full training set.
    df = load_sample(TRAIN_PATH, sample_size=200000)
    #df = load_data(TRAIN_PATH)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    models = get_models()

    results = {}
    for model_name, model in models.items():
        summary = run_cv_experiment(model_name, model, X, y, artifact=ml_aritfact_path)
        results[model_name] = summary

    comparison_df = pd.DataFrame(results).T.sort_values("mean_val_r2", ascending=False)
    print("=== Model comparison (sorted by mean validation R2) ===")
    print(comparison_df[["mean_val_r2", "std_val_r2", "mean_val_mae", "mean_overfit_gap_r2"]])
    comparison_df.to_csv(ml_aritfact_path/"model_comparison_results.csv")
    log_model_comparison(comparison_df, atrifact_path=ml_aritfact_path)

    best_model_name = comparison_df.index[0]
    print(f"\nBest model: {best_model_name} "
          f"(mean val R2 = {comparison_df.loc[best_model_name, 'mean_val_r2']:.4f}, "
          f"overfit gap = {comparison_df.loc[best_model_name, 'mean_overfit_gap_r2']:.4f})")

    with open(ml_aritfact_path/"best_model.json", "w") as f:
        json.dump({"best_model": best_model_name}, f, indent=2)


if __name__ == "__main__":
    main()