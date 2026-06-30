# Deployment

![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Inference-005CED?logo=onnx&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-Supported-4169E1?logo=postgresql&logoColor=white)
![SQL Server](https://img.shields.io/badge/SQL%20Server-Supported-CC2927?logo=microsoftsqlserver&logoColor=white)
![Kafka](https://img.shields.io/badge/Kafka-Optional-231F20?logo=apachekafka&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Runtime-2496ED?logo=docker&logoColor=white)

Deployable application package for the lung disease detection platform. This folder contains the FastAPI runtime, model loading layer, inference pipeline, static frontend bundle, persistence, Kafka integration, Docker files, tests, and MLOps infrastructure.

---

## Table of Contents

- [Runtime Responsibilities](#runtime-responsibilities)
- [Directory Map](#directory-map)
- [Request Lifecycle](#request-lifecycle)
- [Model Runtime Modes](#model-runtime-modes)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Local Run](#local-run)
- [Docker Topologies](#docker-topologies)
- [Database Logging](#database-logging)
- [Prediction Artifacts](#prediction-artifacts)
- [Frontend](#frontend)
- [Testing](#testing)
- [Operational Notes](#operational-notes)

---

## Runtime Responsibilities

The deployment runtime is responsible for:

- validating image input
- loading ONNX or Keras model artifacts
- running segmentation, binary ensemble, and disease classification
- generating prediction artifacts
- returning typed API responses
- logging predictions directly or through Kafka
- exposing the static frontend
- supporting local, Docker, and cloud deployment modes

---

## Directory Map

```text
deployment/
|-- README.md
|-- app/
|   |-- api/              FastAPI app, routes, startup, errors
|   |-- configs/          AppConfig and constants
|   |-- db/               SQLAlchemy models and persistence
|   |-- predictor/        ONNX/Keras model wrappers and pipeline
|   |-- preprocessing/    image loading, ROI, masks, transforms
|   |-- schemas/          Pydantic request and response contracts
|   |-- services/         inference orchestration and output saving
|   +-- utils/            errors, metadata, model loading, visualization
|-- assets/               label mappings and prediction artifact root
|-- db/                   SQL init and migrations
|-- frontend/             compiled static React app
|-- kafka_pipeline/       optional event producer and consumers
|-- mlops/                MLflow, Airflow, retraining, release workflow
|-- reviewed_data/        reviewed-data manifest contract and examples
|-- saved_models/         local model artifacts and metadata
|-- scripts/onnx/         ONNX export and validation utilities
|-- tests/                API, service, unit, contract, MLOps tests
|-- Dockerfile.runtime
|-- docker-compose.runtime.yml
|-- docker-compose.mlops.yml
+-- requirements*.txt
```

---

## Request Lifecycle

```text
POST /predict or /predict/upload
    |
    v
Input validation
    |
    v
Image bytes -> RGB NumPy array -> resized batch
    |
    v
LungDetection.predict()
    |
    |-- segmentation mask
    |-- ROI crop
    |-- binary ensemble
    |-- optional disease classifier
    v
Artifact generation
    |
    |-- source.png
    |-- mask.png
    |-- roi.png
    |-- overlay.png
    v
API response
    |
    +-- direct DB logging, or
    +-- Kafka prediction event
```

---

## Model Runtime Modes

### ONNX Runtime

Default production mode:

```text
MODEL_RUNTIME=onnx
```

Advantages:

- smaller container
- faster cold start than TensorFlow-heavy images
- CPU-friendly inference
- clean Hugging Face Space deployment path

### Keras Runtime

Development and MLflow registry mode:

```text
MODEL_RUNTIME=keras
MLFLOW_ENABLED=true
```

Advantages:

- can load Keras artifacts directly
- can resolve models from MLflow registry
- useful for local debugging and retraining validation

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Runtime health check |
| `GET` | `/` | API identity response |
| `POST` | `/predict` | Predict from JSON image source |
| `POST` | `/predict/upload` | Predict from uploaded file |
| `GET` | `/logs` | Protected prediction log retrieval |
| `GET` | `/ui` | Static frontend entry point |

---

## Configuration

Core configuration lives in `app/configs/config.py`.

| Variable | Default | Notes |
|---|---:|---|
| `MODEL_RUNTIME` | `onnx` | `onnx` or Keras runtime |
| `CORS_ALLOW_ORIGINS` | empty | Comma-separated origin list |
| `DB_LOGGING_ENABLED` | `true` | Enables SQLAlchemy engine setup |
| `DB_BACKEND` | `mssql` | Used when `DATABASE_URL` is not set |
| `DATABASE_URL` | empty | Full connection string override |
| `LOGS_API_KEY` | empty | Enables `/logs` when set |
| `KAFKA_ENABLED` | `true` | Enables prediction event publishing |
| `PREDICTION_STORAGE_BACKEND` | `local` | `local` or `supabase` |
| `HF_MODEL_DOWNLOAD_ENABLED` | `false` | Downloads model artifacts from Hugging Face |

Environment templates:

| File | Purpose |
|---|---|
| `.env.runtime.template` | Slim runtime deployment |
| `.env.compose.template` | Full local stack |
| `.env.hf_spaces.template` | Hugging Face Space style runtime |
| `.env.template` | General local template |

---

## Local Run

```powershell
cd deployment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-test.txt
$env:PYTHONPATH="."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/ui
```

Run tests:

```powershell
$env:PYTHONPATH="."
pytest
```

---

## Docker Topologies

### Runtime Compose

```powershell
docker compose --env-file .env.runtime -f docker-compose.runtime.yml up --build
```

Best for:

- production-style API serving
- ONNX runtime
- local Postgres
- local artifact mounts

### MLOps Compose

```powershell
docker compose --env-file .env.compose -f docker-compose.mlops.yml up --build
```

Best for:

- MLflow registry
- Airflow DAGs
- SQL Server backend
- model evaluation and retraining

---

## Database Logging

Prediction logs are normalized across four tables:

| Table | Description |
|---|---|
| `prediction_requests` | request id, input type, final label, final probability |
| `prediction_binary_model_results` | per-model ensemble outputs |
| `prediction_disease_results` | optional subtype result |
| `prediction_image_links` | generated source/mask/ROI/overlay paths and URLs |

When Kafka is enabled, the DB consumer owns writes. When Kafka is disabled, the API writes directly. This prevents double logging.

---

## Prediction Artifacts

Artifacts are written under date and prediction-id folders:

```text
assets/predictions/YYYY-MM-DD/<prediction-id>/
|-- source.png
|-- mask.png
|-- roi.png
+-- overlay.png
```

The response includes both storage paths and public URLs.

Supported storage backends:

- `local`
- `supabase`

---

## Frontend

`frontend/` contains the compiled static React app. FastAPI mounts it under `/ui` with SPA fallback routing.

Important note:

- The repository currently tracks the built bundle, not the original React source project.
- `frontend/config.js` controls the frontend API base URL and base path behavior.

---

## Testing

The test suite intentionally disables external infrastructure in `tests/conftest.py`.

Coverage areas:

- routes
- inference service behavior
- preprocessing utilities
- ensemble decisions
- metadata contracts
- reviewed-data ingestion
- ONNX conversion smoke test when dependencies exist

Recommended command:

```powershell
cd deployment
$env:PYTHONPATH="."
pytest
```

---

## Operational Notes

- ONNX is the recommended deploy runtime.
- Keras runtime is valuable for registry-driven local validation.
- DICOM is not supported by the current loader.
- Model metadata must remain synchronized with artifacts.
- If `LOGS_API_KEY` is empty, `/logs` is intentionally unavailable.
- If Kafka is unavailable, startup logs a warning and inference can still proceed depending on configuration.
