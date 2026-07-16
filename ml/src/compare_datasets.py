"""
Feature Enrichment Ablation — Random Forest, Original vs Enriched Features
============================================================================
Two MLflow runs are logged — "original_features" and
"enriched_features" — each a 5-fold CV on a stratified sample of the
training set, plus a paired t-test comparing the two runs' fold scores.

This experiment operates on TRAIN ONLY (never val/test), since 5-fold CV
within train already serves the purpose val would otherwise serve here.


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
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import ttest_rel
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from src.config import TRAIN_PATH, TARGET_COL, NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES, ENRICHED_FEATURES, SAMPLE_SIZE, N_SPLITS, RANDOM_STATE

ORIGINAL_FEATURE_SET = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
ENRICHED_FEATURE_SET = ORIGINAL_FEATURE_SET + ENRICHED_FEATURES

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "feature_enrichment_ablation"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sample(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Only rows with full dataset2 enrichment are usable for a fair
    # original-vs-enriched comparison on identical rows (Barley/Cotton
    # rows have NaN ref_* columns and would break the enriched model).
    df = df.dropna(subset=ENRICHED_FEATURES)

    sample = (
        df.groupby("Crop", observed=True, group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), SAMPLE_SIZE // df["Crop"].nunique()),
            random_state=RANDOM_STATE,
        ))
    )
    print(f"Sample loaded with {len(sample)} rows")
    return sample.reset_index(drop=True)


def build_preprocessor(feature_set: list) -> ColumnTransformer:
    numeric_cols = [c for c in NUMERIC_FEATURES + ENRICHED_FEATURES if c in feature_set]
    bool_cols = [c for c in BOOLEAN_FEATURES if c in feature_set]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_set]

    return ColumnTransformer(transformers=[
        ("num", "passthrough", numeric_cols),   # RF is scale-invariant, no need to scale
        ("bool", "passthrough", bool_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def get_parent_feature(col_name: str) -> str:
    """Maps a post-ColumnTransformer column name back to its original
    feature. One-hot dummy columns (e.g. 'cat__Soil_Type_Clay') are mapped
    to their parent categorical ('Soil_Type'), so importance fragmented
    across dummy levels can be summed back into one honest per-feature
    figure. Matched against known categorical names explicitly (rather
    than a generic regex split) since several categorical names themselves
    contain underscores (e.g. 'Soil_Type', 'Weather_Condition'), which
    would break a naive split-on-underscore approach.
    """
    if col_name.startswith("cat__"):
        stripped = col_name[len("cat__"):]
        for cat in CATEGORICAL_FEATURES:
            if stripped.startswith(cat + "_"):
                return cat
        return stripped
    if col_name.startswith("num__"):
        return col_name[len("num__"):]
    if col_name.startswith("bool__"):
        return col_name[len("bool__"):]
    return col_name


# ---------------------------------------------------------------------------
# CV run for one feature set, logged as one MLflow run
# ---------------------------------------------------------------------------

def run_ablation_arm(run_name: str, feature_set: list, X: pd.DataFrame, y: pd.Series, ml_artifact_path:str):
    preprocessor = build_preprocessor(feature_set)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_r2 = []
    fold_rows = []
    fold_importances = []
    last_pipeline = None
    print(f"**** Start {run_name} Cv folding ****")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("ablation_group", "original_vs_enriched")
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
                ("model", RandomForestRegressor(
                    n_estimators=100, max_depth=15,
                    random_state=RANDOM_STATE, n_jobs=2,
                )),
            ])
            pipeline.fit(X_train, y_train)
            val_preds = pipeline.predict(X_val)
            metrics = evaluate(y_val, val_preds)
            # log mae, rmse,r2 for each fold in mlflow
            for k, v in metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=fold_idx)

            fold_r2.append(metrics["r2"])
            fold_rows.append({"fold": fold_idx, **{f"val_{k}": v for k, v in metrics.items()}})

            # Capture this fold's feature importances, keyed by the actual
            # post-encoding column names (so one-hot dummy levels are tracked
            # individually, not just the parent categorical column).
            rf_model = pipeline.named_steps["model"]
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            fold_importances.append(dict(zip(feature_names, rf_model.feature_importances_)))

            last_pipeline = pipeline

            print(f"[{run_name}] fold {fold_idx}: R2={metrics['r2']:.4f}")
        # save 5 folds metrics as html table and logged into mlflow
        fold_df = pd.DataFrame(fold_rows)
        html_path = ml_artifact_path/f"{run_name}_fold_metrics.html"
        fold_df.to_html(html_path, index=False)
        mlflow.log_artifact(html_path)

        mlflow.log_metric("mean_val_r2", float(np.mean(fold_r2)))
        mlflow.log_metric("std_val_r2", float(np.std(fold_r2)))

        # --- Feature importance: mean +/- std across the 5 folds ---------
        # A single split's importances are a noisy snapshot (RF bootstrap
        # sampling + random split selection shift rankings run to run).
        # Aggregating across folds gives a stable ranking and, just as
        # importantly, a variance estimate: features whose importance
        # bounces around fold to fold are not reliable predictors, even if
        # they look decent on any one fold.
        """
        importance_df = pd.DataFrame(fold_importances)
        importance_summary = pd.DataFrame({
            "mean_importance": importance_df.mean(),
            "std_importance": importance_df.std(),
        }).sort_values("mean_importance", ascending=False)

        importance_html_path = ml_artifact_path/f"{run_name}_feature_importance.html"
        importance_summary.to_html(importance_html_path)
        mlflow.log_artifact(importance_html_path)"""

        # --- Grouped importance: one-hot dummies summed back to their ----
        # parent categorical, per fold, THEN averaged across folds. This
        # order matters: summing already-averaged dummy importances would
        # still be a reasonable approximation, but summing within each fold
        # first and averaging the fold-level sums afterward is the
        # statistically correct way to get a mean +/- std for the grouped
        # value, since it preserves the fold-to-fold correlation between a
        # categorical's dummy levels (they're fit together, not independently).
        grouped_fold_importances = []
        for fold_dict in fold_importances:
            grouped = {}
            for col, val in fold_dict.items():
                parent = get_parent_feature(col)
                grouped[parent] = grouped.get(parent, 0.0) + val
            grouped_fold_importances.append(grouped)

        grouped_df = pd.DataFrame(grouped_fold_importances)
        grouped_summary = pd.DataFrame({
            "mean_importance": grouped_df.mean(),
            "std_importance": grouped_df.std(),
        }).sort_values("mean_importance", ascending=False)

        grouped_html_path = ml_artifact_path/f"{run_name}_feature_importance_grouped.html"
        grouped_summary.to_html(grouped_html_path)
        mlflow.log_artifact(grouped_html_path)

        plt.figure(figsize=(9, max(4, len(grouped_summary) * 0.4)))
        plt.barh(
            grouped_summary.index,
            grouped_summary["mean_importance"],
            xerr=grouped_summary["std_importance"],
            color="darkorange",
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Mean Feature Importance (± std across 5 folds)")
        plt.title(f"{run_name}: Feature Importance")
        plt.tight_layout()
        grouped_chart_path = ml_artifact_path/f"{run_name}_feature_importance_grouped.png"
        plt.savefig(grouped_chart_path)
        mlflow.log_artifact(grouped_chart_path)
        plt.close()

        try:
            mlflow.log_table(
                data=grouped_summary.reset_index().rename(columns={"index": "feature"}),
                artifact_file=f"{run_name}_feature_importance_grouped.json",
            )
        except AttributeError:
            grouped_summary.to_csv(f"{run_name}_feature_importance_grouped.csv")
            mlflow.log_artifact(f"{run_name}_feature_importance_grouped.csv")
        """
        try:
            mlflow.log_table(
                data=importance_summary.reset_index().rename(columns={"index": "feature"}),
                artifact_file=f"{run_name}_feature_importance.json",
            )
        except AttributeError:
            importance_summary.to_csv(f"{run_name}_feature_importance.csv")
            mlflow.log_artifact(f"{run_name}_feature_importance.csv")

        plt.figure(figsize=(9, max(4, len(importance_summary) * 0.35)))
        plt.barh(
            importance_summary.index,
            importance_summary["mean_importance"],
            xerr=importance_summary["std_importance"],
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Mean Feature Importance (± std across 5 folds)")
        plt.title(f"{run_name}: Feature Importance (5-fold CV)")
        plt.tight_layout()
        chart_path = f"{run_name}_feature_importance.png"
        plt.savefig(chart_path)
        mlflow.log_artifact(chart_path)
        plt.close()"""

        mlflow.sklearn.log_model(last_pipeline.named_steps["model"], artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    print(f"[{run_name}] mean R2={np.mean(fold_r2):.4f} (+/- {np.std(fold_r2):.4f})\n")
    return fold_r2, run_id


# ---------------------------------------------------------------------------
# Comparison run: paired t-test between the two arms
# ---------------------------------------------------------------------------

def log_comparison(original_scores, enriched_scores, original_run_id, enriched_run_id, ml_atrifact_path:str):
    t_stat, p_value = ttest_rel(enriched_scores, original_scores)

    with mlflow.start_run(run_name="comparison_summary"):
        mlflow.set_tag("ablation_group", "original_vs_enriched")
        mlflow.log_param("compared_runs", json.dumps({
            "original": original_run_id, "enriched": enriched_run_id,
        }))
        mlflow.log_metric("original_mean_r2", float(np.mean(original_scores)))
        mlflow.log_metric("enriched_mean_r2", float(np.mean(enriched_scores)))
        mlflow.log_metric("r2_difference", float(np.mean(enriched_scores) - np.mean(original_scores)))
        mlflow.log_metric("paired_ttest_statistic", float(t_stat))
        mlflow.log_metric("paired_ttest_pvalue", float(p_value))

        conclusion = (
            "No statistically significant improvement from enrichment (p >= 0.05)."
            if p_value >= 0.05 else
            "Statistically significant difference detected (p < 0.05), but check effect size "
            "for practical relevance before treating this as meaningful."
        )
        mlflow.set_tag("conclusion", conclusion)

        summary_df = pd.DataFrame({
            "fold": range(len(original_scores)),
            "original_r2": original_scores,
            "enriched_r2": enriched_scores,
        })
        html_path = ml_atrifact_path/"datasets_comparison.html"
        summary_df.to_html(html_path, index=False)
        mlflow.log_artifact(html_path)

        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(conclusion)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    aritfact_path = BASE_DIR/"ml_artifact"
    print(aritfact_path)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    sample_df = load_sample(TRAIN_PATH)
    y = sample_df[TARGET_COL]

    X_original = sample_df[ORIGINAL_FEATURE_SET].copy()
    X_enriched = sample_df[ENRICHED_FEATURE_SET].copy()

    original_scores, original_run_id = run_ablation_arm("original_features", ORIGINAL_FEATURE_SET, X_original, y, ml_artifact_path=aritfact_path)
    enriched_scores, enriched_run_id = run_ablation_arm("enriched_features", ENRICHED_FEATURE_SET, X_enriched, y, ml_artifact_path=aritfact_path)

    log_comparison(original_scores, enriched_scores, original_run_id, enriched_run_id, ml_atrifact_path=aritfact_path)


if __name__ == "__main__":
    main()