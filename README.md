# AgriTech Answers

[![CI — API](https://github.com/reemHasan/Agritech_Answers/actions/workflows/ci_api.yml/badge.svg)](https://github.com/reemHasan/Agritech_Answers/actions/workflows/ci_api.yml)
[![CD — API](https://github.com/reemHasan/Agritech_Answers/actions/workflows/cd_api.yml/badge.svg)](https://github.com/reemHasan/Agritech_Answers/actions/workflows/cd_api.yml)
[![CD — UI](https://github.com/reemHasan/Agritech_Answers/actions/workflows/cd_ui.yml/badge.svg)](https://github.com/reemHasan/Agritech_Answers/actions/workflows/cd_ui.yml)

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
├── .github/workflows/
│   ├── ci_api.yml                # Test + Docker build validation, on push/PR touching app/backend/**
│   ├── cd_api.yml                # Deploy to Render, gated on ci_api.yml
│   └── cd_ui.yml                 # Deploys frontend to Render on push touching app/frontend/**
│
├── render.yaml                   # Render Blueprint: both services, API_URL auto-linked via fromService
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
Set `API_URL` (env var or `.streamlit/secrets.toml`) to point at the backend — defaults to `http://localhost:8000`.

**Tests:**
```bash
cd app/backend
uv sync
uv run pytest api/test_main.py -v
```
## CI/CD Pipeline Documentation

This project uses three GitHub Actions workflows to automate testing,
building, and deploying both the backend API and the Streamlit frontend.
All workflow files live in `.github/workflows/`.

### Overview

```
   push/PR to app/backend/**              push to app/frontend/**
            │                                        │
            ▼                                        ▼
   ┌─────────────────┐                     ┌──────────────────┐
   │   ci_api.yml    │                     │    cd_ui.yml     │
   │  test → build   │                     │  deploy to Render│
   └────────┬────────┘                     └──────────────────┘
            │ (on success, main only)
            ▼
   ┌───────────────────────────────┐
   │         cd_api.yml            │
   │         deploy (Render)       │
   └───────────────────────────────┘

   Any job failure in ci_api.yml, cd_api.yml, or cd_ui.yml
   triggers a notification by email
```

### Workflows

#### 1. `ci_api.yml` — CI: API tests + build validation

**Triggers:**
- `push` to `main`, only when files under `app/backend/**` change
- `pull_request` targeting `main`, same path filter
- `workflow_dispatch` (manual run from the Actions tab)

**Jobs:**
| Job | What it does |
|---|---|
| `test` | Installs dependencies with `uv`, runs the full `pytest` suite (`app/backend/api/test_main.py`) against a fake model — no real trained artifact needed, so this runs fast and deterministically on every PR. |
| `build` | Builds the API's Docker image (`docker/build-push-action`, `push: false`) to confirm the Dockerfile itself is valid. This is validation only, nothing is pushed here — it exists specifically to catch Dockerfile regressions (a broken `COPY` path, a missing `WORKDIR`, etc.) as early as possible, before they'd otherwise only surface during an actual deployment attempt. |

Both `test` and `build` must pass before `cd_api.yml` will proceed (see below).

#### 2. `cd_api.yml` — CD: deploy API to Render

**Triggers:**
- Automatically via `workflow_run`, once `CI — API` completes successfully on `main`
- `workflow_dispatch` (manual re-run/redeploy without needing a new commit)

**Jobs:**
| Job | What it does |
|---|---|
| `deploy` | POSTs to Render's Deploy Hook for the `crop-yield-api` service, which tells Render to pull the latest commit and rebuild/redeploy the container on its own infrastructure. |


**Why Render deploy hook?**
 The Render deploy hook is what actually makes the live application update; Render builds its own copy of the image server-side from the same Dockerfile.

**Gating on tests**: `cd_api.yml` only runs after `ci_api.yml` succeeds (via `workflow_run`), and Render's own "Auto-Deploy" setting is disabled (`autoDeploy: false` in `render.yaml`) — so a broken test or a broken Docker build blocks deployment entirely, rather than Render deploying on every raw push regardless of test outcome.

#### 3. `cd_ui.yml` — CD: deploy the Streamlit frontend

**Triggers:**
- `push` to `main`, only when files under `app/frontend/**` change
- `workflow_dispatch`

**Jobs:**
| Job | What it does |
|---|---|
| `deploy` | POSTs to Render's Deploy Hook for the `crop-yield-ui` service. |

No dedicated test suite runs here (the frontend contains no business/ML logic — it's a thin client over the already-tested API — so a smoke-test-only pipeline was judged sufficient rather than adding a second test suite).

### How to read pipeline status

The badges at the top of the `README.md` reflect each workflow's most
recent run on `main` — green means the last run passed, red means it
failed. Click a badge to jump straight to that workflow's run history in
the Actions tab.

### Local equivalents

To reproduce what CI does, locally:

```bash
# What ci_api.yml's `test` job does:
cd app/backend
uv sync
uv run pytest api/test_main.py -v

# What ci_api.yml's `build` job does:
docker build -f app/backend/Dockerfile -t crop-yield-api:local ./app/backend
```

## Deployment (Render)

Two Render Web Services, both defined in a single root-level `render.yaml`
Blueprint, each built from its own Dockerfile:

| Service | Dockerfile | Context | Deploy trigger |
|---|---|---|---|
| `crop-yield-api` | `app/backend/Dockerfile` | `app/backend` | `cd_api.yml`, after `ci_api.yml` tests pass |
| `crop-yield-ui` | `app/frontend/Dockerfile` | `app/frontend` | `cd_ui.yml`, on push to `app/frontend/**` |

`crop-yield-ui`'s `API_URL` env var is auto-linked to `crop-yield-api`'s
deployed hostname via `fromService` — no manual URL copy-paste needed
after redeploys.

Both services have Render's own "Auto-Deploy" setting turned **off**
(`autoDeploy: false`) — deploys only happen via the CI/CD workflows' deploy
hooks, not on every raw push, so the API's deploy stays gated on tests
passing.

<!--**Adopting the Blueprint** (if migrating from manually-created services,
as this project initially was): `render.yaml` is only read by Render's
**New → Blueprint** flow, not **New → Web Service** — a service created
manually never looks at this file at all. To adopt it: delete the manually
created services in the Render dashboard, then **New → Blueprint**,
connect the repo (root `render.yaml` is auto-detected, no custom path
needed), and deploy. Re-add the two deploy hook secrets afterward, since
they're per-service and change when services are recreated.
-->

### Required secrets & repository variables

Configure under **Repo Settings → Secrets and variables → Actions**:

| Name | Type | Used by | Purpose |
|---|---|---|---|
| `RENDER_API_DEPLOY_HOOK_URL` | Secret | `cd_api.yml` | Triggers the API's Render deploy |
| `RENDER_UI_DEPLOY_HOOK_URL` | Secret | `cd_ui.yml` | Triggers the UI's Render deploy |

## Producing the model artifact

`app/backend/model/ridge_pipeline.joblib` is produced by
`ml/src/utils_app.py`, which will export the trained and registred pipeline in MLflow register, then copy it into `app/backend/model/`.