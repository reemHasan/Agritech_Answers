"""
Final Model Training & Evaluation — Ridge (Production Candidate)
====================================================================
Refits the selected, tuned final model (Ridge, alpha=1.0, solver="auto" --
confirmed via randomized search to be robust across alpha 0.001-3.5) on the
FULL training set (not a sample), then formally evaluates it once on
val.csv and once on test.csv.

test.csv is touched exactly once, at the very end, as the final unbiased
performance estimate

The fitted pipeline (preprocessing + model) is logged to MLflow and
registered in the Model Registry, ready to be loaded by the prediction/
recommendation API.

"""

import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from src.config import TRAIN_PATH, TARGET_COL, NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES, VAL_PATH, TEST_PATH

ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# Final selected hyperparameters, confirmed via randomized search
# (25 trials, alpha 0.001-100): R2 flat 0.001-3.5, alpha=1.0 chosen for
# numerical robustness at no cost to accuracy.
RIDGE_ALPHA = 1.0
RIDGE_SOLVER = "auto"
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "crop_yield_final_model"
REGISTERED_MODEL_NAME = "crop_yield_ridge_model"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df


# ---------------------------------------------------------------------------
# Preprocessing
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


def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

def _parent_feature(col_name: str) -> str:
    """Maps a post-ColumnTransformer column name back to its original
    feature (e.g. 'cat__Soil_Type_Clay' -> 'Soil_Type'), so one-hot
    dummies collapse back into one human-readable bar per category
    instead of fragmenting across many small dummy-level bars."""
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
# Main
# ---------------------------------------------------------------------------

def main(output_path:str):
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df = load_data(TRAIN_PATH)
    val_df = load_data(VAL_PATH)
    test_df = load_data(TEST_PATH)

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET_COL]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET_COL]
    X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET_COL]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    with mlflow.start_run(run_name="ridge_final_production_model"):
        mlflow.log_param("model_type", "ridge")
        mlflow.log_param("alpha", RIDGE_ALPHA)
        mlflow.log_param("solver", RIDGE_SOLVER)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_val_rows", len(X_val))
        mlflow.log_param("n_test_rows", len(X_test))
        mlflow.log_param("features", json.dumps(ALL_FEATURES))

        # --- Fit on the FULL training set -----------------------------
        preprocessor = build_preprocessor()
        model = Ridge(alpha=RIDGE_ALPHA, solver=RIDGE_SOLVER, random_state=42)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        # --- Training-set metrics (reference point only, not for selection)
        train_preds = pipeline.predict(X_train)
        train_metrics = evaluate(y_train, train_preds)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        print(f"Train:      R2={train_metrics['r2']:.4f}  MAE={train_metrics['mae']:.4f}  "
              f"RMSE={train_metrics['rmse']:.4f}")

        # --- Validation metrics -----------------------------------------
        val_preds = pipeline.predict(X_val)
        val_metrics = evaluate(y_val, val_preds)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        print(f"Validation: R2={val_metrics['r2']:.4f}  MAE={val_metrics['mae']:.4f}  "
              f"RMSE={val_metrics['rmse']:.4f}")

        # --- Test metrics: FINAL, ONE-TIME evaluation --------------------
        # This is the only point in the entire project where test.csv is
        # used. Its result is the unbiased estimate of real-world
        # performance and should be reported as-is, regardless of outcome.
        test_preds = pipeline.predict(X_test)
        test_metrics = evaluate(y_test, test_preds)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        print(f"Test:       R2={test_metrics['r2']:.4f}  MAE={test_metrics['mae']:.4f}  "
              f"RMSE={test_metrics['rmse']:.4f}")

        # --- Overfit / generalization gap diagnostics ---------------------
        mlflow.log_metric("overfit_gap_train_val_r2", train_metrics["r2"] - val_metrics["r2"])
        mlflow.log_metric("generalization_gap_val_test_r2", val_metrics["r2"] - test_metrics["r2"])

        # --- Coefficients, for interpretability / the business report -----
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": pipeline.named_steps["model"].coef_,
        }).assign(abs_coef=lambda d: d["coefficient"].abs()).sort_values("abs_coef", ascending=False)

        coef_html = output_path/"final_model_coefficients.html"
        coef_df.drop(columns="abs_coef").to_html(coef_html, index=False)
        mlflow.log_artifact(coef_html)
        print("\nTop coefficients (raw, one-hot dummies not grouped):")
        print(coef_df.drop(columns="abs_coef").head(10).to_string(index=False))
        # --- Grouped coefficients: one-hot dummies collapsed to parent -----
        # IMPORTANT: for a fully one-hot-encoded categorical (all levels,
        # no drop='first') combined with an intercept, Ridge's L2 penalty
        # picks the minimum-norm solution among infinitely many equally
        # valid ones -- and that solution has the dummy coefficients of
        # each categorical group sum to ~0 BY CONSTRUCTION, regardless of
        # whether that categorical actually matters. Summing is therefore
        # meaningless for these groups. The meaningful question is "how
        # much does switching between levels move the prediction" -- i.e.
        # the RANGE (max - min) of the coefficients within the group, not
        # their sum. Numeric/boolean features map to a single raw column
        # each (no such redundancy), so their "grouped" value is just that
        # one coefficient, unchanged.
        #
        # NOTE: this mixes two different statistics in one column --
        # numeric/boolean rows are a true SIGNED coefficient (direction
        # matters), while categorical rows are a non-negative RANGE
        # (direction doesn't apply, only magnitude). Kept in one table for
        # a single "which feature matters most" ranking, but the "type"
        # column below makes the distinction explicit so it's never
        # mistaken for a like-for-like comparison.
        grouped_raw = {}
        for name, coef in zip(coef_df["feature"], coef_df["coefficient"]):
            parent = _parent_feature(name)
            grouped_raw.setdefault(parent, []).append(coef)

        grouped_coef, grouped_type = {}, {}
        for parent, values in grouped_raw.items():
            if len(values) == 1:
                grouped_coef[parent] = values[0]
                grouped_type[parent] = "coefficient"
            else:
                grouped_coef[parent] = max(values) - min(values)
                grouped_type[parent] = "range (max-min)"

        grouped_coef_df = pd.DataFrame({
            "feature": list(grouped_coef.keys()),
            "coefficient": list(grouped_coef.values()),
            "type": [grouped_type[f] for f in grouped_coef.keys()],
        }).assign(abs_coef=lambda d: d["coefficient"].abs()).sort_values("abs_coef", ascending=False)

        grouped_html = output_path/"final_model_coefficients_grouped.html"
        grouped_coef_df.drop(columns="abs_coef").to_html(grouped_html, index=False)
        mlflow.log_artifact(grouped_html)
        print("\nGrouped coefficients (numeric/boolean: signed coefficient; categorical: range across levels):")
        print(grouped_coef_df.drop(columns="abs_coef").to_string(index=False))

        # --- Coefficient plot: sorted by magnitude, colored by type --------
        # Numeric/boolean rows are a true signed coefficient (red=negative,
        # green=positive relationship with yield). Categorical rows are a
        # non-negative RANGE across levels (see note above) -- coloring
        # these green/red would falsely imply a "positive/negative
        # relationship", which doesn't apply to a range. Given a distinct
        # neutral color (blue) instead, so the two statistic types are
        # visually distinguishable, not just distinguishable in a table.
        coef_plot_df = grouped_coef_df.sort_values("abs_coef", ascending=True)  # ascending for horizontal barh top-down
        colors = [
            "#3a6ea5" if t.startswith("range") else ("#d62728" if c < 0 else "#2ca02c")
            for c, t in zip(coef_plot_df["coefficient"], coef_plot_df["type"])
        ]

        plt.figure(figsize=(9, max(4, len(coef_plot_df) * 0.35)))
        plt.barh(coef_plot_df["feature"], coef_plot_df["coefficient"], color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Numeric/boolean: signed coefficient (red/green)  |  Categorical: range across levels (blue)")
        plt.title("Ridge Coefficients — Grouped by Feature")
        plt.tight_layout()
        coef_chart_path = output_path/"final_model_coefficients.png"
        plt.savefig(coef_chart_path, dpi=150)
        mlflow.log_artifact(coef_chart_path)
        plt.close()

        # --- Zoomed-in version: excludes the top 3 dominant features -------
        # Fertilizer_Used / Rainfall_mm / Irrigation_Used sit an order of
        # magnitude above everything else, which flattens all the minor
        # features into an indistinguishable line on a shared x-axis. This
        # second view drops the top 3 so the remaining features' relative
        # differences (still tiny in absolute terms, but not all identical)
        # are visible -- useful to confirm none of the "minor" features is
        # secretly larger than the others once the dominant ones are removed.
        top_n_excluded = 3
        excluded_features = grouped_coef_df.sort_values("abs_coef", ascending=False).head(top_n_excluded)["feature"].tolist()
        zoomed_df = coef_plot_df[~coef_plot_df["feature"].isin(excluded_features)]
        zoomed_colors = [
            "#3a6ea5" if t.startswith("range") else ("#d62728" if c < 0 else "#2ca02c")
            for c, t in zip(zoomed_df["coefficient"], zoomed_df["type"])
        ]

        plt.figure(figsize=(9, max(4, len(zoomed_df) * 0.35)))
        plt.barh(zoomed_df["feature"], zoomed_df["coefficient"], color=zoomed_colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Numeric/boolean: signed coefficient (red/green)  |  Categorical: range across levels (blue)")
        plt.title(f"Ridge Coefficients — Zoomed In (excludes top {top_n_excluded}: "
                  f"{', '.join(excluded_features)})")
        plt.tight_layout()
        coef_zoomed_chart_path = output_path/"final_model_coefficients_zoomed.png"
        plt.savefig(coef_zoomed_chart_path, dpi=150)
        mlflow.log_artifact(coef_zoomed_chart_path)
        plt.close()

        # --- Log and register the full pipeline (preprocessing + model) ---
        # Registering the whole pipeline, not just the bare Ridge model,
        # so the API can call .predict() directly on raw feature input
        # without needing to reimplement scaling/encoding separately.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\nModel registered as '{REGISTERED_MODEL_NAME}', run_id={run_id}")

    # --- Summary file for downstream steps (API packaging, report) --------
    summary = {
        "run_id": run_id,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alpha": RIDGE_ALPHA,
        "solver": RIDGE_SOLVER,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(output_path/"final_model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ml_artifact_folder = base_dir/"ml_artifact"/"final_model_eval3"
    ml_artifact_folder.mkdir(parents=True, exist_ok=True)
    main(ml_artifact_folder)