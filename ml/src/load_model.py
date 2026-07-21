import joblib
import mlflow
from pathlib import Path

# Replace with your run_id
run_id = "de0e2c6c326f480d8d84794ae21cc68f"
model_name ="model"
# Load model from MLflow
model = mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")

# Save as Joblib
base_dir = Path(__file__).resolve().parent.parent
ml_artifact_folder = base_dir/"model"
ml_artifact_folder.mkdir(parents=True, exist_ok=True)
joblib.dump(model, ml_artifact_folder/"ridge_pipeline.joblib")

print(f"Model saved to {ml_artifact_folder}/ridge_pipeline.joblib")