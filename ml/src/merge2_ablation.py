"""
Track 2 Ablation — Native vs Dataset1-Enriched Features on Dataset2
=======================================================================
Mirrors merge1_ablation.py in the reversed merge direction: tests whether
dataset1-derived crop-level aggregates (rainfall, temperature,
fertilizer/irrigation rate, days to harvest) improve prediction of
real-world yield (dataset2), using BOTH Ridge and CatBoost via 5-fold CV.

Only the 4 crops with a genuine dataset1 match (Wheat, Rice, Maize,
Soybeans) are used, so both arms are compared on identical rows -- the
other dataset2 crops (Cassava, Plantains and others, Potatoes, Sweet
potatoes, Yams, and any others without a dataset1 counterpart) are
excluded from this specific ablation.

Each (model, feature_set) combination is logged as its own MLflow run,
plus a per-model paired t-test comparing native vs enriched.

"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.catboost
from mlflow import MlflowClient
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import ttest_rel
from pathlib import Path
from catboost import CatBoostRegressor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent.parent
MERGED_DATASET_PATH = DATA_PATH/"data/processed_data/reversed_crop_dataset.csv"
TARGET_COL = "yield_tons_per_ha"

NATIVE_NUMERIC = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
NATIVE_CATEGORICAL = ["Area", "Item"]
NATIVE_FEATURE_SET = NATIVE_NUMERIC + NATIVE_CATEGORICAL

ENRICHED_NUMERIC = [
    "d1_ref_rainfall_mm", "d1_ref_temperature_c",
    "d1_ref_fertilizer_rate", "d1_ref_irrigation_rate", "d1_ref_days_to_harvest",
]
ENRICHED_FEATURE_SET = NATIVE_FEATURE_SET + ENRICHED_NUMERIC

N_SPLITS = 5
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "track2_ablation_ridge_catboost"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Only rows with a genuine dataset1 match, so both ablation arms are
    # compared on identical rows (mirrors the original ablation's approach).
    matched = df.dropna(subset=ENRICHED_NUMERIC).reset_index(drop=True)
    print(f"Matched rows (Wheat/Rice/Maize/Soybeans): {len(matched)} of {len(df)} total")
    print("Missing vaules:",matched.isnull().sum())
    return matched

# ---------------------------------------------------------------------------
# Preprocessing — per model, per feature set
# ---------------------------------------------------------------------------

def build_preprocessor(model_name: str, feature_set: list) -> ColumnTransformer:
    numeric_cols = [c for c in NATIVE_NUMERIC + ENRICHED_NUMERIC if c in feature_set]
    cat_cols = [c for c in NATIVE_CATEGORICAL if c in feature_set]

    numeric_transformer = Pipeline(steps=[
        #("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    if model_name == "catboost":
        categorical_transformer = Pipeline(steps=[
            ("to_str", FunctionTransformer(lambda X: X.astype(str), feature_names_out="one-to-one")),
        ])
    else:
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols),
    ])


def catboost_categorical_indices(feature_set: list) -> list:
    numeric_cols = [c for c in NATIVE_NUMERIC + ENRICHED_NUMERIC if c in feature_set]
    cat_cols = [c for c in NATIVE_CATEGORICAL if c in feature_set]
    return list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))


def build_model(model_name: str, feature_set: list):
    if model_name == "ridge":
        return Ridge(alpha=RIDGE_ALPHA, solver="auto", random_state=RANDOM_STATE)
    if model_name == "catboost":
        return CatBoostRegressor(
            iterations=500, random_state=RANDOM_STATE,
            cat_features=catboost_categorical_indices(feature_set), verbose=False,
        )
    raise ValueError(model_name)


def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# CV run for one (model, feature_set) combination
# ---------------------------------------------------------------------------

def run_ablation_arm(model_name: str, run_name: str, feature_set: list,
                      X: pd.DataFrame, y: pd.Series, output_path:str):
    preprocessor = build_preprocessor(model_name, feature_set)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_r2 = []
    fold_rows = []
    fold_importances = []  # feature importance (catboost) or coefficients (ridge)
    last_pipeline = None

    with mlflow.start_run(run_name=f"{model_name}_{run_name}"):
        mlflow.set_tag("ablation_group", "track2_native_vs_enriched")
        mlflow.set_tag("model_family", model_name)
        mlflow.log_param("feature_set", run_name)
        mlflow.log_param("n_features", len(feature_set))
        mlflow.log_param("n_rows", len(X))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = build_model(model_name, feature_set)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            metrics = evaluate(y_val, val_preds)

            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=fold_idx)

            fold_r2.append(metrics["r2"])
            fold_rows.append({"fold": fold_idx, **{f"val_{k}": v for k, v in metrics.items()}})

            # Capture this fold's feature importance (CatBoost) or
            # coefficients (Ridge), keyed by the actual post-encoding
            # column names.
            fold_model = pipeline.named_steps["model"]
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            if model_name == "catboost":
                values = fold_model.get_feature_importance()
            else:
                values = fold_model.coef_
            fold_importances.append(dict(zip(feature_names, values)))

            last_pipeline = pipeline
            print(f"[{model_name}/{run_name}] fold {fold_idx}: R2={metrics['r2']:.4f}")

        fold_df = pd.DataFrame(fold_rows)
        html_path = output_path/f"{model_name}_{run_name}_fold_metrics.html"
        fold_df.to_html(html_path, index=False)
        mlflow.log_artifact(html_path)

        mlflow.log_metric("mean_val_r2", float(np.mean(fold_r2)))
        mlflow.log_metric("std_val_r2", float(np.std(fold_r2)))

        # --- Mean +/- std feature importance / coefficients across folds ---
        importance_label = "Feature Importance" if model_name == "catboost" else "Coefficient"
        importance_df = pd.DataFrame(fold_importances)
        importance_summary = pd.DataFrame({
            "mean_value": importance_df.mean(),
            "std_value": importance_df.std(),
        })
        importance_summary["abs_mean"] = importance_summary["mean_value"].abs()
        importance_summary = importance_summary.sort_values("abs_mean", ascending=False)

        importance_html = output_path/f"{model_name}_{run_name}_importance.html"
        importance_summary.drop(columns="abs_mean").to_html(importance_html)
        mlflow.log_artifact(importance_html)

        try:
            mlflow.log_table(
                data=importance_summary.drop(columns="abs_mean").reset_index().rename(columns={"index": "feature"}),
                artifact_file=f"{model_name}_{run_name}_importance.json",
            )
        except AttributeError:
            importance_summary.drop(columns="abs_mean").to_csv(f"{model_name}_{run_name}_importance.csv")
            mlflow.log_artifact(f"{model_name}_{run_name}_importance.csv")

        plot_df = importance_summary.sort_values("abs_mean", ascending=True)
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in plot_df["mean_value"]] \
            if model_name == "ridge" else None  # CatBoost importances are non-negative by nature

        plt.figure(figsize=(9, max(4, len(plot_df) * 0.3)))
        plt.barh(plot_df.index, plot_df["mean_value"], xerr=plot_df["std_value"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel(f"Mean {importance_label} (± std across {N_SPLITS} folds)")
        plt.title(f"{model_name} / {run_name}: {importance_label}")
        plt.tight_layout()
        chart_path = output_path/f"{model_name}_{run_name}_importance.png"
        plt.savefig(chart_path, dpi=150)
        mlflow.log_artifact(chart_path)
        plt.close()

        if model_name == "catboost":
            mlflow.catboost.log_model(last_pipeline.named_steps["model"], artifact_path="model")
        else:
            mlflow.sklearn.log_model(last_pipeline.named_steps["model"], artifact_path="model")
        run_id = mlflow.active_run().info.run_id

    print(f"[{model_name}/{run_name}] mean val R2={np.mean(fold_r2):.4f} "
          f"(+/- {np.std(fold_r2):.4f})\n")
    return fold_r2, run_id, importance_summary.drop(columns="abs_mean")


# ---------------------------------------------------------------------------
# Comparison run, per model family
# ---------------------------------------------------------------------------

def log_comparison(model_name: str, native_scores, enriched_scores, native_run_id, enriched_run_id):
    t_stat, p_value = ttest_rel(enriched_scores, native_scores)

    with mlflow.start_run(run_name=f"{model_name}_comparison_summary"):
        mlflow.set_tag("ablation_group", "track2_native_vs_enriched")
        mlflow.set_tag("model_family", model_name)
        mlflow.log_param("compared_runs", json.dumps({
            "native": native_run_id, "enriched": enriched_run_id,
        }))
        mlflow.log_metric("native_mean_r2", float(np.mean(native_scores)))
        mlflow.log_metric("enriched_mean_r2", float(np.mean(enriched_scores)))
        mlflow.log_metric("r2_difference", float(np.mean(enriched_scores) - np.mean(native_scores)))
        mlflow.log_metric("paired_ttest_statistic", float(t_stat))
        mlflow.log_metric("paired_ttest_pvalue", float(p_value))

        conclusion = (
            f"[{model_name}] No statistically significant improvement from enrichment (p >= 0.05)."
            if p_value >= 0.05 else
            f"[{model_name}] Statistically significant difference detected (p < 0.05)."
        )
        mlflow.set_tag("conclusion", conclusion)
        print(f"[{model_name}] paired t-test: t={t_stat:.4f}, p={p_value:.4f} -- {conclusion}")

    return {"model": model_name, "t_stat": t_stat, "p_value": p_value,
            "native_r2": np.mean(native_scores), "enriched_r2": np.mean(enriched_scores)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_dir = Path(__file__).resolve().parent.parent
    ml_artifact_folder = base_dir/"ml_artifact"/"trak2_ablation"
    ml_artifact_folder.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    matched_df =  load_data(MERGED_DATASET_PATH)
    y = matched_df[TARGET_COL]
    X_native = matched_df[NATIVE_FEATURE_SET].copy()
    X_enriched = matched_df[ENRICHED_FEATURE_SET].copy()

    summary_rows = []
    for model_name in ["ridge", "catboost"]:
        native_scores, native_run_id, native_importance = run_ablation_arm(
            model_name, "native_features", NATIVE_FEATURE_SET, X_native, y, output_path=ml_artifact_folder)
        enriched_scores, enriched_run_id, enriched_importance = run_ablation_arm(
            model_name, "enriched_features", ENRICHED_FEATURE_SET, X_enriched, y, output_path=ml_artifact_folder)
        result = log_comparison(model_name, native_scores, enriched_scores,
                                 native_run_id, enriched_run_id)
        summary_rows.append(result)

        print(f"\n[{model_name}] enriched-features importance for d1_ref_* columns:")
        ref_rows = enriched_importance[enriched_importance.index.str.contains("d1_ref")]
        print(ref_rows)

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Track 2 ablation summary (Ridge & CatBoost) ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(ml_artifact_folder/"track2_ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()