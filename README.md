---
title: Crop Yield Prediction API
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Crop Yield Prediction API

FastAPI service exposing crop yield prediction and recommendation endpoints,
backed by a tuned Ridge regression model. See `/docs` for interactive API
documentation once deployed.

- `POST /predict` — predicts yield for one chosen crop under given parcel conditions.
- `POST /recommend` — ranks all known crops by predicted yield for given parcel conditions.

<!-- pytest --cov=app/ --cov-report html -->
