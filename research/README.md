# Research

![TensorFlow](https://img.shields.io/badge/TensorFlow-Training-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Model%20Experiments-D00000?logo=keras&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Search-3155A4)
![TFRecord](https://img.shields.io/badge/TFRecord-Data%20Pipeline-orange)
![Status](https://img.shields.io/badge/Role-Research%20Artifacts-lightgrey)

Research notebooks, dataset preparation utilities, TFRecord generation workflows, and training experiments for the lung disease detection platform.

---

## Table of Contents

- [Purpose](#purpose)
- [Dataset Classes](#dataset-classes)
- [Research Tracks](#research-tracks)
- [Folder Structure](#folder-structure)
- [Utility Modules](#utility-modules)
- [How Research Connects to Deployment](#how-research-connects-to-deployment)
- [Reproducibility Notes](#reproducibility-notes)
- [Notebook Policy](#notebook-policy)

---

## Purpose

The `research` folder preserves the experimental history behind the deployable models. It is intentionally separate from the FastAPI runtime so notebooks, exploratory plots, architecture searches, and training utilities do not leak into production serving code.

The research layer answers:

- how the dataset was indexed and transformed
- how TFRecords were generated
- how segmentation and classification models were trained
- how Optuna searches informed DenseNet configurations
- how class mappings evolved into deployment contracts

---

## Dataset Classes

The source dataset uses four classes:

| Source Label | Numeric Label | Deployment Usage |
|---|---:|---|
| `COVID` | `0` | Unhealthy subtype |
| `Normal` | `1` | Healthy binary class |
| `Viral Pneumonia` | `2` | Unhealthy subtype |
| `Lung_Opacity` | `3` | Unhealthy subtype |

Deployment remaps this into:

| Task | Labels |
|---|---|
| Binary classification | `Healthy`, `Unhealthy` |
| Disease classification | `COVID`, `Viral Pneumonia`, `Lung Opacity` |
| Segmentation | `background`, `lung` |

---

## Research Tracks

### Data Preparation

| Notebook | Purpose |
|---|---|
| `create_initial_files.ipynb` | Builds metadata CSVs and mapping files |
| `create_tfrecords.ipynb` | Serializes images, masks, and labels into TFRecord shards |

### Binary Classification

Located under `binary-healthy_unhealthy/`.

Architectures explored:

- DenseNet121
- EfficientNetV2B3
- InceptionV3
- MobileNetV3

The deployable system uses these as a four-model ensemble.

### Disease Classification

Located under `multiclass-diseases/`.

The disease classifier is trained only on unhealthy classes:

- COVID
- Viral Pneumonia
- Lung Opacity

This matches the deployment design, where subtype classification runs only after an unhealthy binary decision.

### Segmentation

Located under `segmentation/`.

The segmentation notebooks train and validate a U-Net style model with an Xception encoder. Its output mask supports:

- ROI cropping
- mask artifacts
- overlay artifacts
- mask-aware classifier preprocessing

### Optuna

Optuna notebooks under DenseNet experiment folders preserve:

- architecture search results
- optimization hyperparameter search results
- best parameter JSON files
- SQLite study databases

The MLOps backfill workflow can log selected notebook and Optuna artifacts into MLflow.

---

## Folder Structure

```text
research/
|-- README.md
|-- data/
|   |-- README.md.txt
|   |-- all_image_mask_pairs.csv
|   |-- all_metadata.csv
|   |-- class_mapping.json
|   |-- *_mapping.json
|   +-- image and mask folders
|-- images/
|   +-- architecture and documentation assets
|-- binary-healthy_unhealthy/
|   |-- EfficientNetV2B3-healthy-binary_classification.ipynb
|   |-- InceptionV3-healthy-binary_classification.ipynb
|   |-- MobileNetV3-healthy-binary_classification.ipynb
|   +-- densenet/
|-- multiclass-diseases/
|   |-- DenseNet121-diseases-multiclass_classification.ipynb
|   |-- utils.py
|   +-- optuna/
|-- segmentation/
|   |-- segemntation-U-Net_Xception.ipynb
|   +-- segmentation-checking_multitask_model.ipynb
|-- create_initial_files.ipynb
|-- create_tfrecords.ipynb
+-- utils.py
```

---

## Utility Modules

| File | Purpose |
|---|---|
| `research/utils.py` | Shared TensorFlow parsing, augmentation, dataset, ROI, cleanup, metrics, and pruning helpers |
| `binary-healthy_unhealthy/densenet/optuna/utils.py` | Binary DenseNet Optuna helper functions |
| `multiclass-diseases/utils.py` | Disease classification dataset and model helpers |
| `multiclass-diseases/optuna/utils.py` | Disease DenseNet Optuna helper functions |

These files are the only research Python modules intended for shared reuse. Notebooks remain experiment artifacts.

---

## How Research Connects to Deployment

Research outputs become deployment inputs through:

```text
Research notebooks
    |
    |-- trained Keras models
    |-- ONNX exports
    |-- metadata.yaml
    |-- class mapping JSON
    |-- TFRecord files
    v
deployment/
    |
    |-- FastAPI runtime
    |-- MLOps evaluation and retraining jobs
    |-- model registry and release workflows
```

The critical bridge is `metadata.yaml`. It tells the runtime how to preprocess inputs, where the model artifacts live, which threshold to use, and what labels to expose.

---

## Reproducibility Notes

The research utilities include helpers for:

- seeding TensorFlow, NumPy, and Python randomness
- selecting TPU, GPU, or CPU distribution strategy
- parsing TFRecords consistently
- preserving binary lung masks with nearest-neighbor resizing
- applying synchronized image-mask augmentation
- using ROI crops with anatomical context margins
- calculating class weights and steps from datasets
- cleaning TensorFlow sessions between experiments

Notebook results still depend on the training hardware, TensorFlow version, dataset snapshot, and random seeds.

---

## Notebook Policy

Notebooks are intentionally not treated as production code. They should:

- document experiments and training decisions
- preserve plots, metrics, and architecture notes
- call shared helpers from utility modules where possible
- export models and metadata for deployment

Production runtime behavior should live under `deployment/`, not inside notebooks.
