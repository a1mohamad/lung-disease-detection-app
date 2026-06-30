# MLOps

[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%7C%20Registry-0194E2)](https://mlflow.org/)
[![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Retraining-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Release%20Validation-005CED?logo=onnx&logoColor=white)](https://onnx.ai/)

**Deployment Targets**

[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Model%20Publishing-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/a1mohamadd/lung-disease-detection)
[![Render PostgreSQL](https://img.shields.io/badge/Render-Model%20Logs%20DB-46E3B7?logo=render&logoColor=black)](https://render.com/docs/databases)
[![Supabase](https://img.shields.io/badge/Supabase-Reviewed%20Data%20%7C%20Artifacts-3FCF8E?logo=supabase&logoColor=white)](https://supabase.com/storage)
[![Research Lab](https://img.shields.io/badge/Research%20Lab-Project%20Page-222222?logo=githubpages&logoColor=white)](https://a1mohamad.github.io/research/lung-disease-detection/index.html)

Evaluation, retraining, tracking, registry, release, and reviewed-data ingestion workflows for the lung disease detection platform.

---

## Table of Contents

- [Purpose](#purpose)
- [Managed Models](#managed-models)
- [Workflow Overview](#workflow-overview)
- [Folder Structure](#folder-structure)
- [Dataset Modes](#dataset-modes)
- [Reviewed Data Ingestion](#reviewed-data-ingestion)
- [Evaluation](#evaluation)
- [Retraining](#retraining)
- [Promotion and Release](#promotion-and-release)
- [Airflow DAGs](#airflow-dags)
- [MLflow Logging](#mlflow-logging)
- [Commands](#commands)
- [Safety Guarantees](#safety-guarantees)

---

## Purpose

The MLOps layer exists to make model updates measurable, repeatable, and auditable. It handles:

- monthly model evaluation
- retraining on legacy or reviewed-data snapshots
- MLflow experiment logging
- registry version creation
- promotion decisions based on metrics
- Keras release staging
- ONNX export and numerical validation
- optional Hugging Face publishing
- Airflow orchestration

In the public deployment, promoted artifacts are designed to be synchronized with Hugging Face Hub so the Hugging Face Space runtime can download the same validated ONNX/Keras artifacts that were produced by the release workflow.

Research context and experiment presentation are published in the [lung disease detection research lab](https://a1mohamad.github.io/research/lung-disease-detection/index.html).

---

## Managed Models

| Spec | Task | Registered Model | Promotion Metric |
|---|---|---|---|
| `densenet_binary` | Binary classification | `lung-binary-densenet` | `val_f1` |
| `efficientnet_binary` | Binary classification | `lung-binary-efficientnet` | `val_f1` |
| `inception_binary` | Binary classification | `lung-binary-inception` | `val_f1` |
| `mobilenet_binary` | Binary classification | `lung-binary-mobilenet` | `val_f1` |
| `densenet_diseases` | Disease classification | `lung-diseases-densenet` | `val_f1` |
| `unet_xception_segmentation` | Segmentation | `lung-segmentation-unet-xception` | `dice_coefficient` |

Model specs are defined in `config/model_specs.py`.

---

## Workflow Overview

```text
Airflow schedule or manual CLI
    |
    v
Resolve dataset mode
    |
    |-- legacy TFRecords
    +-- prepared reviewed-data snapshot
    |
    v
Build tf.data pipeline
    |
    v
Load model from MLflow registry or local artifact
    |
    v
Evaluate or retrain
    |
    v
Log metrics, params, metadata, signature, summary
    |
    v
Register model version
    |
    v
Promotion decision
    |
    v
Stage Keras + ONNX release
    |
    v
Optional Hugging Face publication
```

---

## Folder Structure

```text
mlops/
|-- README.md
|-- airflow/
|   |-- dags/
|   +-- tasks/
|-- config/
|   |-- model_specs.py
|   +-- settings.py
|-- core/
|   |-- backfill/
|   |-- data/
|   |-- evaluation/
|   |-- features/
|   |-- ingestion/
|   |-- models/
|   |-- publishing/
|   |-- tracking/
|   +-- training/
|-- jobs/
|   |-- monthly_log_results.py
|   |-- monthly_retrain.py
|   +-- post_hoc_backfill.py
|-- mlflow/
|-- requirements-mlops.txt
+-- requirements-onnx-exporter.txt
```

---

## Dataset Modes

### Legacy Mode

```text
RETRAIN_DATASET_MODE=legacy
```

Uses TFRecord files already present in the research dataset folder. Files are split into train and validation subsets by ratio.

Best for:

- reproducing earlier experiments
- quick local evaluation
- baseline model comparison

### Prepared Mode

```text
RETRAIN_DATASET_MODE=prepared
```

Builds a formal snapshot from reviewed manifests and optional baseline data. The snapshot contains:

```text
snapshot-YYYYMMDDTHHMMSSZ/
|-- train/
|-- validation/
|-- test/
|-- manifest.json
+-- records.jsonl
```

Best for:

- production-style retraining
- reviewed data governance
- stable patient-level splits
- auditable dataset snapshots

---

## Reviewed Data Ingestion

Reviewed-data manifests are validated before they enter training.

Validation includes:

- schema version
- batch id
- period start/end
- record timestamps
- supported class names
- safe object keys
- duplicate sample ids
- optional SHA-256 checksums
- patient-level split consistency

The ingestion layer can read from:

- local filesystem
- Supabase Storage

---

## Evaluation

Evaluation jobs:

- load model metadata
- build task-specific validation datasets
- load model from MLflow or local artifacts
- compute metrics
- log metadata and summaries to MLflow

Task-specific metrics:

| Task | Metrics |
|---|---|
| Binary classification | accuracy, precision, recall, F1, AUC |
| Multiclass classification | accuracy, macro precision, macro recall, macro F1 |
| Segmentation | Dice coefficient, IoU |

---

## Retraining

Retraining jobs:

- build train/validation datasets
- compute steps from TFRecords
- fine-tune models with early stopping
- evaluate validation metrics
- optionally evaluate test metrics for prepared snapshots
- log metrics and artifacts
- optionally register a new model version

Manual retraining is intentionally guarded. Use `--allow-manual-run` when running outside Airflow.

---

## Promotion and Release

Promotion compares a candidate run metric to the current production alias metric.

If the candidate improves:

1. the model can receive the MLflow `production` alias
2. release artifacts can be staged
3. ONNX export is created
4. ONNX output is validated against Keras output
5. artifacts can be published to Hugging Face Hub

Release metadata is written as `release.json` beside staged artifacts.

Configured live model destination:

```text
HF_MODEL_REPO_ID=a1mohamadd/lung-disease-detection
HF_MODEL_REPO_TYPE=model
```

---

## Airflow DAGs

| DAG | Purpose |
|---|---|
| `log_models_monthly` | monthly evaluation logging |
| `orchestrate_retrain_pipeline` | prepares dataset and triggers all retraining DAGs |
| `retrain_binary_densenet` | retrains DenseNet binary model |
| `retrain_binary_efficientnet` | retrains EfficientNet binary model |
| `retrain_binary_inception` | retrains Inception binary model |
| `retrain_binary_mobilenet` | retrains MobileNet binary model |
| `retrain_diseases_densenet` | retrains disease classifier |
| `retrain_segmentation_unet_xception` | retrains segmentation model |

---

## MLflow Logging

Runs can log:

- task and model tags
- training parameters
- metadata YAML
- reported metrics from metadata
- fresh evaluation metrics
- Keras model artifacts
- model signatures
- release artifacts
- run summary JSON
- notebook and Optuna artifacts for backfill runs

---

## Commands

Monthly evaluation:

```powershell
$env:PYTHONPATH="."
python -m mlops.jobs.monthly_log_results
```

Retrain one model:

```powershell
$env:PYTHONPATH="."
python -m mlops.jobs.monthly_retrain `
  --model-name densenet_binary `
  --allow-manual-run
```

Post-hoc backfill:

```powershell
$env:PYTHONPATH="."
python -m mlops.jobs.post_hoc_backfill --with-eval
```

Full stack:

```powershell
docker compose --env-file .env.compose -f docker-compose.mlops.yml up --build
```

---

## Safety Guarantees

- Patient-level split registry prevents patient leakage across train, validation, and test.
- Snapshot fingerprinting detects mismatched repeated snapshots.
- Prepared retraining raises on release failures instead of silently drifting.
- BatchNormalization layers can remain frozen during fine-tuning workflows.
- ONNX releases are validated numerically before publication.
- Manual retraining is blocked unless explicitly allowed.
