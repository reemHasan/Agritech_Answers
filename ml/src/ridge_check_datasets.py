"""
Final Model Sanity Check — Ridge, Original vs Enriched Features
===================================================================
Repeats the original-vs-enriched ablation using the actual selected final
model (Ridge, alpha=1.0 — confirmed via randomized search to be robust
across alpha 0.001-3.5), rather than the Random Forest used in the earlier
exploratory ablation. Also tracks per-fold coefficients, so the enriched
variables' influence can be inspected directly rather than only through
aggregate R²/MAE — a diagnostic Random Forest's feature_importances_
couldn't offer as precisely (linear coefficients are directly interpretable
in the model's own units, without one-hot fragmentation ambiguity for
numeric features).

"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import ttest_rel
from pathlib import Path



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from src.config import TRAIN_PATH, TARGET_COL, NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES, ENRICHED_FEATURES, SAMPLE_SIZE, N_SPLITS, RANDOM_STATE

# ref_yield_tons_per_ha intentionally excluded from both sets: it is a
# yield-derived proxy and would constitute leakage if used as a predictor.

ORIGINAL_FEATURE_SET = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
ENRICHED_FEATURE_SET = ORIGINAL_FEATURE_SET + ENRICHED_FEATURES

RIDGE_ALPHA = 1.0  # confirmed via randomized search: robust across 0.001-3.5
RIDGE_SOLVER = "auto"

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "final_model_feature_ablation"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sample(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Only rows with full dataset2 enrichment are usable for a fair
    # original-vs-enriched comparison on identical rows.
    df = df.dropna(subset=ENRICHED_FEATURES)

    sample = (
        df.groupby("Crop", observed=True, group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), SAMPLE_SIZE // df["Crop"].nunique()),
            random_state=RANDOM_STATE,
        ))
    )
    return sample.reset_index(drop=True)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ENRICHED_FEATURE_SET + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df.dropna(subset=ENRICHED_FEATURES)

def build_preprocessor(feature_set: list) -> ColumnTransformer:
    numeric_cols = [c for c in NUMERIC_FEATURES + ENRICHED_FEATURES if c in feature_set]
    bool_cols = [c for c in BOOLEAN_FEATURES if c in feature_set]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_set]

    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("bool", "passthrough", bool_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# CV run for one feature set, logged as one MLflow run
# ---------------------------------------------------------------------------

def run_ablation_arm(run_name: str, feature_set: list, X: pd.DataFrame, y: pd.Series, artifact_path:str):
    preprocessor = build_preprocessor(feature_set)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_r2 = []
    fold_rows = []
    fold_coefs = []
    last_pipeline = None

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("ablation_group", "final_model_original_vs_enriched")
        mlflow.log_param("model_type", "ridge")
        mlflow.log_param("alpha", RIDGE_ALPHA)
        mlflow.log_param("solver", RIDGE_SOLVER)
        mlflow.log_param("feature_set", run_name)
        mlflow.log_param("n_features", len(feature_set))
        mlflow.log_param("features", json.dumps(feature_set))
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("sample_size", len(X))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(alpha=RIDGE_ALPHA, solver=RIDGE_SOLVER, random_state=RANDOM_STATE)),
            ])
            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            metrics = evaluate(y_val, val_preds)

            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=fold_idx)

            # Capture this fold's coefficients, keyed by post-encoding
            # column name (so each one-hot dummy is tracked individually,
            # and each numeric/enriched feature's own coefficient is exact
            ridge_model = pipeline.named_steps["model"]
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            fold_coefs.append(dict(zip(feature_names, ridge_model.coef_)))

            fold_r2.append(metrics["r2"])
            fold_rows.append({"fold": fold_idx, **{f"val_{k}": v for k, v in metrics.items()}})
            last_pipeline = pipeline

            print(f"[{run_name}] fold {fold_idx}: R2={metrics['r2']:.4f}")

        fold_df = pd.DataFrame(fold_rows)
        html_path = artifact_path/f"{run_name}_fold_metrics.html"
        fold_df.to_html(html_path, index=False)
        mlflow.log_artifact(html_path)

        mlflow.log_metric("mean_val_r2", float(np.mean(fold_r2)))
        mlflow.log_metric("std_val_r2", float(np.std(fold_r2)))

        # --- Coefficients: mean +/- std across the 5 folds -----------
        coef_df = pd.DataFrame(fold_coefs)
        coef_summary = pd.DataFrame({
            "mean_coef": coef_df.mean(),
            "std_coef": coef_df.std(),
            "abs_mean_coef": coef_df.mean().abs(),
        }).sort_values("abs_mean_coef", ascending=False)

        coef_html_path = artifact_path/f"{run_name}_coefficients.html"
        coef_summary.to_html(coef_html_path)
        mlflow.log_artifact(coef_html_path)

        plt.figure(figsize=(9, max(4, len(coef_summary) * 0.35)))
        plt.barh(coef_summary.index, coef_summary["mean_coef"], xerr=coef_summary["std_coef"])
        plt.gca().invert_yaxis()
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Mean Ridge Coefficient (± std across 5 folds)")
        plt.title(f"{run_name}: Ridge Coefficients (standardized features)")
        plt.tight_layout()
        coef_chart_path = artifact_path/f"{run_name}_coefficients.png"
        plt.savefig(coef_chart_path, dpi=150)
        mlflow.log_artifact(coef_chart_path)
        plt.close()

        mlflow.sklearn.log_model(last_pipeline.named_steps["model"], artifact_path="model")
        run_id = mlflow.active_run().info.run_id

    print(f"[{run_name}] mean R2={np.mean(fold_r2):.4f} (+/- {np.std(fold_r2):.4f})\n")
    return fold_r2, run_id, coef_summary


# ---------------------------------------------------------------------------
# Comparison run: paired t-test + coefficient overlay for the enriched terms
# ---------------------------------------------------------------------------

def log_comparison(original_scores, enriched_scores, original_run_id, enriched_run_id,
                    enriched_coef_summary, artifact_path:str):
    t_stat, p_value = ttest_rel(enriched_scores, original_scores)

    with mlflow.start_run(run_name="comparison_summary"):
        mlflow.set_tag("ablation_group", "final_model_original_vs_enriched")
        mlflow.log_param("compared_runs", json.dumps({
            "original": original_run_id, "enriched": enriched_run_id,
        }))
        mlflow.log_metric("original_mean_r2", float(np.mean(original_scores)))
        mlflow.log_metric("enriched_mean_r2", float(np.mean(enriched_scores)))
        mlflow.log_metric("r2_difference", float(np.mean(enriched_scores) - np.mean(original_scores)))
        mlflow.log_metric("paired_ttest_statistic", float(t_stat))
        mlflow.log_metric("paired_ttest_pvalue", float(p_value))

        # Isolate just the ref_* coefficients from the enriched run's summary
        ref_coefs = enriched_coef_summary[
            enriched_coef_summary.index.str.contains("ref_")
        ]
        mlflow.log_metrics({
            f"coef_{name.replace('num__', '')}": row["mean_coef"]
            for name, row in ref_coefs.iterrows()
        })

        conclusion = (
            "No statistically significant improvement from enrichment (p >= 0.05)."
            if p_value >= 0.05 else
            "Statistically significant difference detected (p < 0.05) -- check effect "
            "size and ref_* coefficient magnitudes below for practical relevance."
        )
        mlflow.set_tag("conclusion", conclusion)

        summary_df = pd.DataFrame({
            "fold": range(len(original_scores)),
            "original_r2": original_scores,
            "enriched_r2": enriched_scores,
        })
        html_path = artifact_path/"ridge_ablation_comparison.html"
        summary_df.to_html(html_path, index=False)
        mlflow.log_artifact(html_path)

        ref_html_path = artifact_path/"ridge_enriched_ref_coefficients.html"
        ref_coefs.to_html(ref_html_path)
        mlflow.log_artifact(ref_html_path)

        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(conclusion)
        print("\nref_* coefficients (enriched model):")
        print(ref_coefs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    ml_aritfact_path = BASE_DIR/"ml_artifact"
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    data = load_data(TRAIN_PATH)
    #sample_df = load_sample(TRAIN_PATH)
    y = data[TARGET_COL]

    X_original = data[ORIGINAL_FEATURE_SET].copy()
    X_enriched = data[ENRICHED_FEATURE_SET].copy()

    original_scores, original_run_id, _ = run_ablation_arm(
        "original_features_ridge_compare", ORIGINAL_FEATURE_SET, X_original, y, artifact_path=ml_aritfact_path)
    enriched_scores, enriched_run_id, enriched_coef_summary = run_ablation_arm(
        "enriched_features_ridge_compare", ENRICHED_FEATURE_SET, X_enriched, y, artifact_path=ml_aritfact_path)

    log_comparison(original_scores, enriched_scores, original_run_id, enriched_run_id,
                    enriched_coef_summary, artifact_path=ml_aritfact_path)


if __name__ == "__main__":
    main()