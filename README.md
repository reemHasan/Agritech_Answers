# AgriTech Answers

Crop yield prediction and recommendation system for farmers: a FastAPI
backend serving a tuned Ridge regression model, and a Streamlit frontend
for interactive use.

## Structure

```
agritech_answers/
│
├── app/
│   ├── backend/                 # FastAPI service (deployed to Render)
│   │   ├── api/
│   │   │   ├── main.py              # App, routes, lifespan/model loading wiring
│   │   │   ├── pydantic_models.py   # Request/response schemas, Enums
│   │   │   ├── logger.py            # Structured JSON logging setup
│   │   │   ├── helpers.py           # Model loading, prediction logic
│   │   │   └── test_main.py         # Pytest suite (uses a fake model, no real artifact needed)
│   │   ├── model/
│   │   │   └── ridge_pipeline.joblib   # Trained pipeline artifact (see ml/src/train_final_model.py)
│   │   ├── Dockerfile
│   │   ├── render.yaml
│   │   ├── pyproject.toml           # Lean runtime deps only (no mlflow, no jupyter)
│   │   └── uv.lock
│   │
│   └── frontend/                # Streamlit UI (deployed to Render as a 2nd service)
│       ├── app.py
│       ├── Dockerfile
│       ├── requirements.txt
│       └── .streamlit/
│           ├── config.toml
│           └── secrets.toml.example
│
├── ml/
│   ├── notebook/                # Jupyter notebooks (EDA, PCA/FAMD, merge exploration)
│   └── src/                     # Training pipeline scripts
│       ├── train_model_comparison.py     # Ridge/LinearRegression/LightGBM/CatBoost/RF, 5-fold CV
│       ├── tune_ridge.py                 # Randomized search on Ridge's alpha/solver
│       ├── train_final_model.py          # Refits on full train, evals val/test, registers in MLflow
│       ├── mlflow_feature_ablation.py    # Original-vs-enriched ablation (Random Forest)
│       ├── final_model_ablation_check.py # Same ablation, using the final tuned Ridge
│       ├── track1_ablation.py            # Ridge + CatBoost ablation, dataset1 primary
│       ├── track2_ablation.py            # Ridge + CatBoost ablation, dataset2 primary (reversed merge)
│       ├── reverse_merge_comparison.py   # 5-model comparison on the reversed merge
│       ├── reverse_merge_ablation.py     # CatBoost ablation on the reversed merge
│       └── compute_field_bounds.py       # Generates ui_options.json for the Streamlit sliders/dropdowns
│
├── data/                        # Raw/intermediate datasets (not committed if large -- see .gitignore)
│
├── reports/
│   ├── data_merge_summary.md         # Merge strategy + key variables (French)
│   ├── resume_fusion_donnees.md      # Same, alternate framing
│   └── project_summary_report.md     # Full project summary: EDA -> modeling -> both merge directions
│
├── .github/workflows/
│   ├── ci_api.yml                # Tests the backend on push/PR touching app/backend/**
│   ├── cd_api.yml                # Deploys backend to Render, gated on ci_api.yml succeeding
│   └── cd_ui.yml                 # Deploys frontend to Render on push touching app/frontend/**
│
├── pyproject.toml                # Root: full research/dev environment (notebooks, MLOps tooling)
├── uv.lock
└── README.md
```

## Local development

**Backend:**
```bash
cd app/backend
uv sync
cd api
uv run --project .. uvicorn main:app --reload
```

**Frontend** (in a second terminal):
```bash
cd app/frontend
pip install -r requirements.txt
streamlit run app.py
```
Set `API_URL` (env var or `.streamlit/secrets.toml`, copy from `secrets.toml.example`) to point at the backend — defaults to `http://localhost:8000`.

**Tests:**
```bash
cd app/backend
uv sync
uv run pytest api/test_main.py -v
```

## Deployment (Render)

Two independent Render Web Services, each built from its own Dockerfile:

| Service | Dockerfile | Context | Deploy trigger |
|---|---|---|---|
| `crop-yield-api` | `app/backend/Dockerfile` | `app/backend` | `cd_api.yml`, after `ci_api.yml` tests pass |
| `crop-yield-ui` | `app/frontend/Dockerfile` | `app/frontend` | `cd_ui.yml`, on push to `app/frontend/**` |

Both services have Render's own "Auto-Deploy" setting turned **off**
(`autoDeploy: false` in `render.yaml`) — deploys only happen via the CI/CD
workflows' deploy hooks, not on every raw push, so the API's deploy stays
gated on tests passing.

**Required GitHub repo secrets:**
- `RENDER_API_DEPLOY_HOOK_URL`
- `RENDER_UI_DEPLOY_HOOK_URL`

(Render Dashboard → each service → Settings → Deploy Hook)

## Producing the model artifact

`app/backend/model/ridge_pipeline.joblib` is produced by
`ml/src/train_final_model.py`, then exported with `joblib.dump(pipeline,
"ridge_pipeline.joblib")` and copied into `app/backend/model/` before
building the backend Docker image.
<!-- pytest --cov=app/ --cov-report html -->
