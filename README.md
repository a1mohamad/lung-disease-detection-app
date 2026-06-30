<div align="center">

# Lung Disease Detection

### End-to-end chest X-ray analysis platform with FastAPI, ONNX inference, Kafka, MLflow, Airflow, and Hugging Face deployment

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?logo=fastapi&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Production%20Inference-005CED?logo=onnx&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Training%20%7C%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![Kafka](https://img.shields.io/badge/Kafka-Event%20Pipeline-231F20?logo=apachekafka&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Tracking-0194E2)
![Airflow](https://img.shields.io/badge/Airflow-MLOps%20Orchestration-017CEE?logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Why This Project Matters](#why-this-project-matters)
- [System Capabilities](#system-capabilities)
- [Model Pipeline](#model-pipeline)
- [Architecture](#architecture)
- [Application Modules](#application-modules)
- [Repository Structure](#repository-structure)
- [API Reference](#api-reference)
- [Prediction Response](#prediction-response)
- [Runtime Configuration](#runtime-configuration)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [MLOps Lifecycle](#mlops-lifecycle)
- [Kafka Event Pipeline](#kafka-event-pipeline)
- [Testing and CI/CD](#testing-and-cicd)
- [Data and Model Artifacts](#data-and-model-artifacts)
- [Responsible AI and Clinical Safety](#responsible-ai-and-clinical-safety)
- [Portfolio Talking Points](#portfolio-talking-points)
- [Current Limitations](#current-limitations)
- [License](#license)

---

## Overview

**Lung Disease Detection** is a portfolio-grade machine learning system for chest X-ray analysis. It is built as a complete ML product rather than a single notebook: research experiments, trained model artifacts, a production API, generated visual artifacts, event-driven logging, MLOps orchestration, model release tooling, and cloud deployment are all represented in the repository.

The system predicts:

| Stage | Output | Purpose |
|---|---|---|
| **Lung segmentation** | Binary lung mask | Isolate lung region and produce visual artifacts |
| **Binary ensemble** | Healthy / Unhealthy | First-level screening decision |
| **Disease classifier** | COVID, Viral Pneumonia, Lung Opacity | Subtype prediction for unhealthy scans |

The deployable runtime is optimized around **ONNX Runtime** for lean inference, while the research and MLOps layers retain **TensorFlow/Keras**, **MLflow**, and **Airflow** for training, evaluation, registry, and release workflows.

---

## Why This Project Matters

This repository demonstrates the full arc of an applied AI system:

- Research notebooks are separated from deployment code.
- Model metadata defines thresholds, preprocessing, labels, and artifact paths.
- Inference is exposed through a documented API.
- Prediction outputs include human-reviewable visual artifacts.
- Events can be logged directly or published to Kafka.
- MLOps jobs evaluate, retrain, register, promote, export, validate, and publish models.
- Tests cover service logic, preprocessing contracts, model metadata, ONNX release behavior, and reviewed-data ingestion.

For a resume flagship, the important point is the **engineering completeness**: this is not only model training; it is an ML system with runtime, observability hooks, reproducibility, and deployment boundaries.

---

## System Capabilities

### Inference

- Accepts images by local path, public URL, base64 payload, or multipart upload.
- Normalizes image inputs to RGB and resizes to `256 x 256`.
- Runs segmentation before classification.
- Uses an ensemble of four binary classifiers.
- Runs disease subtype classification only when the binary ensemble predicts an unhealthy scan.
- Generates source, mask, ROI, and overlay artifacts.
- Supports local artifact serving and Supabase-backed storage.

### API and Frontend

- FastAPI application with typed Pydantic request and response schemas.
- Static React frontend bundle mounted under `/ui`.
- Health endpoint for CI/CD and cloud runtime checks.
- Protected prediction log endpoint.

### Eventing and Persistence

- Direct database logging when Kafka is disabled.
- Kafka event publishing when Kafka is enabled.
- Independent consumers for database persistence, analytics, monitoring, doctor-review queues, and notifications.
- SQL Server and Postgres support through SQLAlchemy.

### MLOps

- MLflow experiment logging and registry support.
- Airflow DAGs for monthly evaluation and retraining.
- Reviewed-data ingestion with manifest validation.
- Stable patient-level train/validation/test splits.
- ONNX export and numerical validation.
- Optional Hugging Face Hub publication for promoted model releases.

---

## Model Pipeline

```text
Input X-ray
    |
    v
Image validation + RGB resize
    |
    v
U-Net Xception segmentation
    |
    |-- mask artifact
    |-- overlay artifact
    v
Lung ROI extraction
    |
    v
Binary ensemble
    |-- DenseNet121
    |-- EfficientNetV2B3
    |-- InceptionV3
    |-- MobileNetV3
    |
    v
Final healthy/unhealthy decision
    |
    +-- Healthy: return binary result
    |
    +-- Unhealthy:
            |
            v
        DenseNet disease classifier
            |
            v
        COVID / Viral Pneumonia / Lung Opacity
```

### Model Metadata Contract

Every deployment model directory includes a `metadata.yaml` file. Metadata controls:

- model name and task
- Keras artifact path
- ONNX artifact path
- preprocessing strategy
- input size and channel count
- classification threshold
- output labels
- reported metrics

This makes runtime behavior auditable and avoids hard-coding model-specific details throughout the application.

---

## Architecture

```text
                            +-----------------------+
                            |  Static React UI      |
                            |  /ui                  |
                            +-----------+-----------+
                                        |
                                        v
+------------------+        +-----------+-----------+
| Client / cURL    +-------> | FastAPI Inference API |
+------------------+        +-----------+-----------+
                                        |
                                        v
                            +-----------+-----------+
                            | LungDetection Pipeline |
                            | segmentation + ensemble|
                            +-----------+-----------+
                                        |
                      +-----------------+-----------------+
                      |                                   |
                      v                                   v
          +-----------+-----------+          +------------+------------+
          | Prediction Artifacts  |          | Prediction Response     |
          | source/mask/roi/over  |          | labels/probs/links      |
          +-----------+-----------+          +------------+------------+
                      |                                   |
                      v                                   v
          +-----------+-----------+          +------------+------------+
          | Local or Supabase     |          | DB logging or Kafka     |
          | artifact storage      |          | event publication       |
          +-----------------------+          +------------+------------+
                                                           |
                                                           v
                                             +-------------+-------------+
                                             | Kafka Consumers           |
                                             | DB / analytics / monitor  |
                                             | doctor queue / notify     |
                                             +---------------------------+
```

---

## Application Modules

| Module | Responsibility |
|---|---|
| `deployment/app/api` | FastAPI routes, startup lifecycle, and exception mapping |
| `deployment/app/services` | Request-level inference orchestration, input selection, artifact saving |
| `deployment/app/predictor` | Runtime model wrappers and the high-level lung detection pipeline |
| `deployment/app/preprocessing` | Image loading, ROI extraction, normalization, mask handling, metadata-driven transforms |
| `deployment/app/db` | SQLAlchemy models, session setup, prediction log persistence |
| `deployment/app/schemas` | Pydantic contracts for requests, responses, logs, and health checks |
| `deployment/app/utils` | Metadata loading, model loading, ONNX wrapper, metrics, visualization, errors |
| `deployment/kafka_pipeline` | Prediction event builder, producer, and consumers |
| `deployment/mlops` | Evaluation, retraining, reviewed-data ingestion, MLflow tracking, release, and Airflow DAGs |
| `research` | Training notebooks, dataset preparation utilities, Optuna experiments, TFRecord generation |

---

## Repository Structure

```text
Lung Disease Detection/
|-- README.md
|-- LICENSE
|-- .github/
|   +-- workflows/
|       +-- ci.yml
|-- research/
|   |-- README.md
|   |-- data/
|   |-- binary-healthy_unhealthy/
|   |-- multiclass-diseases/
|   |-- segmentation/
|   |-- create_initial_files.ipynb
|   |-- create_tfrecords.ipynb
|   +-- utils.py
|-- deployment/
|   |-- README.md
|   |-- app/
|   |-- assets/
|   |-- db/
|   |-- frontend/
|   |-- kafka_pipeline/
|   |-- mlops/
|   |-- reviewed_data/
|   |-- saved_models/
|   |-- scripts/
|   |-- tests/
|   |-- Dockerfile.runtime
|   |-- docker-compose.runtime.yml
|   |-- docker-compose.mlops.yml
|   |-- requirements-runtime.txt
|   |-- requirements-test.txt
|   +-- requirements.txt
```

---

## API Reference

### Health

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Predict from JSON

```http
POST /predict?return_all=true
Content-Type: application/json
```

Supported JSON inputs:

```json
{"image_path": "C:/path/to/xray.png"}
```

```json
{"image_url": "https://example.com/xray.png"}
```

```json
{"image_base64": "data:image/png;base64,..."}
```

Exactly one image source must be provided.

### Predict from Upload

```http
POST /predict/upload?return_all=true
Content-Type: multipart/form-data
```

Example:

```bash
curl -X POST "http://localhost:8000/predict/upload?return_all=true" \
  -F "file=@xray.png"
```

### Logs

```http
GET /logs?limit=50&offset=0
X-Api-Key: <LOGS_API_KEY>
```

The logs endpoint is disabled unless `LOGS_API_KEY` is configured.

---

## Prediction Response

```json
{
  "final_prob": 0.84,
  "final_probs_by_label": {
    "healthy": 0.16,
    "unhealthy": 0.84
  },
  "final_label": 1,
  "final_label_name": "Unhealthy",
  "models_results": {
    "densenet": {
      "prob": 0.91,
      "probs_by_label": {
        "healthy": 0.09,
        "unhealthy": 0.91
      },
      "label": 1,
      "label_name": "unhealthy"
    }
  },
  "disease": {
    "label": 0,
    "label_name": "COVID",
    "probs_by_label": {
      "COVID": 0.72,
      "Viral Pneumonia": 0.18,
      "Lung Opacity": 0.10
    }
  },
  "source_url": "/static/predictions/2026-06-29/<id>/source.png",
  "mask_url": "/static/predictions/2026-06-29/<id>/mask.png",
  "roi_url": "/static/predictions/2026-06-29/<id>/roi.png",
  "overlay_url": "/static/predictions/2026-06-29/<id>/overlay.png"
}
```

---

## Runtime Configuration

| Variable | Default | Description |
|---|---:|---|
| `MODEL_RUNTIME` | `onnx` | Selects ONNX or Keras runtime |
| `DB_LOGGING_ENABLED` | `true` | Enables prediction database logging |
| `DB_BACKEND` | `mssql` | Uses SQL Server or Postgres when `DATABASE_URL` is not set |
| `DATABASE_URL` | empty | Full SQLAlchemy connection string override |
| `KAFKA_ENABLED` | `true` | Publishes prediction events to Kafka |
| `KAFKA_BOOTSTRAP_SERVERS` | `127.0.0.1:9092` | Kafka broker list |
| `MLFLOW_ENABLED` | `false` | Enables MLflow model registry loading for Keras runtime |
| `HF_MODEL_DOWNLOAD_ENABLED` | `false` | Downloads missing model artifacts from Hugging Face |
| `PREDICTION_STORAGE_BACKEND` | `local` | `local` or `supabase` artifact storage |
| `LOGS_API_KEY` | empty | Enables and protects `/logs` |

Runtime templates are provided in `deployment/.env*.template`.

---

## Local Development

From the deployment folder:

```powershell
cd deployment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-test.txt
```

Run tests:

```powershell
$env:PYTHONPATH="."
pytest
```

Run the API:

```powershell
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/ui
```

---

## Docker Deployment

### Slim ONNX Runtime

This is the production-oriented topology for API serving:

```powershell
cd deployment
docker compose --env-file .env.runtime -f docker-compose.runtime.yml up --build
```

Includes:

- FastAPI app
- ONNX Runtime inference
- optional local Postgres service
- mounted model and artifact folders

### Full MLOps Stack

```powershell
cd deployment
docker compose --env-file .env.compose -f docker-compose.mlops.yml up --build
```

Includes:

- SQL Server
- MLflow
- Airflow
- retraining jobs
- model registry workflow

---

## MLOps Lifecycle

```text
Reviewed data / baseline data
    |
    v
Validated manifest ingestion
    |
    v
Stable patient-level splits
    |
    v
TFRecord snapshot
    |
    v
Airflow retraining DAG
    |
    v
MLflow run + metrics + model artifact
    |
    v
Promotion decision
    |
    v
Keras release + ONNX export + validation
    |
    v
Optional Hugging Face Hub publication
```

Promotion is metric-driven:

| Task | Promotion Metric |
|---|---|
| Binary classifiers | `val_f1` |
| Disease classifier | `val_f1` |
| Segmentation model | `dice_coefficient` |

Prepared retraining mode is designed to prevent production drift: when a prepared model is promoted, the system can require Hugging Face publication so the registry and runtime artifact source remain synchronized.

---

## Kafka Event Pipeline

When Kafka is enabled, the API publishes a `prediction.completed` event instead of writing directly to the DB.

Consumers:

| Consumer | Responsibility |
|---|---|
| `consumer_db.py` | Persist normalized prediction logs |
| `consumer_analytics.py` | Append compact analytics JSONL records |
| `consumer_monitoring.py` | Maintain rolling five-minute operational metrics |
| `consumer_doctor_images.py` | Build an artifact queue for clinical review workflows |
| `consumer_notifications.py` | Create user-facing notification messages |

This design keeps inference responsive and allows downstream workflows to evolve independently.

---

## Testing and CI/CD

The test suite covers:

- API route behavior
- request validation
- inference response shaping
- preprocessing contracts
- ensemble aggregation
- metadata and class mapping contracts
- ONNX release smoke validation
- reviewed-data manifest and split logic

Run locally:

```powershell
cd deployment
$env:PYTHONPATH="."
pytest
```

GitHub Actions performs:

1. dependency installation
2. pytest with coverage
3. runtime Docker build on push
4. Hugging Face Space deployment
5. deployed `/health` verification

---

## Data and Model Artifacts

The research dataset is organized around:

- COVID
- Normal
- Viral Pneumonia
- Lung Opacity
- lung masks

Deployment label contracts:

```json
{
  "0": "Healthy",
  "1": "Unhealthy"
}
```

```json
{
  "0": "COVID",
  "1": "Viral Pneumonia",
  "2": "Lung Opacity"
}
```

Model artifacts are expected under:

```text
deployment/saved_models/
|-- healthy_unhealthy/
|-- diseases/
+-- segmentation/
```

Each model directory contains `metadata.yaml`, Keras artifacts, and ONNX artifacts.

---

## Responsible AI and Clinical Safety

This repository is for research, education, and portfolio demonstration. It is not a certified medical device.

Important safety points:

- Predictions must not be used as a standalone clinical diagnosis.
- Chest X-ray interpretation requires qualified medical review.
- Dataset bias, acquisition differences, hospital-specific protocols, and scanner variation can affect predictions.
- Generated masks and overlays are explainability aids, not proof of correctness.
- Any real clinical deployment would require external validation, monitoring, governance, and regulatory review.

---

## Portfolio Talking Points

This project demonstrates:

- end-to-end ML product design
- clean research-to-deployment separation
- metadata-driven inference
- ONNX optimization for production
- FastAPI API design
- typed request/response contracts
- SQLAlchemy persistence with multiple database backends
- Kafka event-driven architecture
- MLflow registry and promotion logic
- Airflow orchestration
- reviewed-data ingestion and stable splitting
- ONNX export validation
- Dockerized deployment
- CI/CD to Hugging Face Spaces

---

## Current Limitations

- The frontend source is not included; only the built static bundle is tracked.
- Model artifacts are large and may need external hosting for clean public distribution.
- The project is CPU-oriented in runtime Docker configuration.
- Clinical robustness has not been externally validated.
- DICOM input is explicitly rejected by the current image loader.
- The Kafka consumers use simple JSONL outputs for some downstream workflows rather than full production services.

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
