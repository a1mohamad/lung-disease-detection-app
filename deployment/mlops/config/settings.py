"""Environment-driven settings for MLOps jobs and Airflow tasks."""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable using common truthy strings.

    Args:
        name: Environment variable name.
        default: Boolean used when the variable is absent.

    Returns:
        Parsed boolean value.
    """
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class MLOpsSettings:
    """Centralized configuration for data, tracking, release, and publishing jobs.

    The MLOps layer uses this class instead of the API ``AppConfig`` because
    training jobs need additional paths and safety switches: reviewed-data
    ingestion, split registries, snapshot output, ONNX release validation, and
    optional Hugging Face publication.
    """

    # Resolve both "running from deployment/" and "running from repository root"
    # layouts so Airflow, local scripts, and CI can share the same code.
    ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
    DEPLOYMENT_DIR = ROOT
    if not (DEPLOYMENT_DIR / "saved_models").exists() and (
        ROOT / "deployment" / "saved_models"
    ).exists():
        DEPLOYMENT_DIR = ROOT / "deployment"

    RESEARCH_DIR = ROOT / "research"
    if not RESEARCH_DIR.exists() and (ROOT.parent / "research").exists():
        RESEARCH_DIR = ROOT.parent / "research"

    # Legacy research TFRecords are still supported, but prepared reviewed-data
    # snapshots are the safer path for production retraining.
    TFRECORDS_DIR = RESEARCH_DIR / "data" / "tfrecords"
    EXPERIMENT = "lung-detection"
    MODEL_STAGE = "Production"

    BATCH_SIZE = 16
    EPOCHS = 20
    VAL_RATIO = 0.2
    MAX_TRAIN_BATCHES = None
    MAX_EVAL_BATCHES = None

    # Reviewed-data settings define where new human-reviewed examples are read
    # from before they are converted into immutable TFRecord snapshots.
    RETRAIN_DATASET_MODE = os.getenv("RETRAIN_DATASET_MODE", "legacy").strip().lower()
    REVIEWED_DATA_BACKEND = os.getenv("REVIEWED_DATA_BACKEND", "local").strip().lower()
    REVIEWED_DATA_LOCAL_ROOT = Path(
        os.getenv(
            "REVIEWED_DATA_LOCAL_ROOT",
            str(DEPLOYMENT_DIR / "reviewed_data" / "incoming"),
        )
    )
    REVIEWED_DATA_MANIFEST_INDEX = os.getenv(
        "REVIEWED_DATA_MANIFEST_INDEX",
        "index.json",
    ).strip("/")
    REVIEWED_DATA_INCLUDE_BASELINE = _env_bool(
        "REVIEWED_DATA_INCLUDE_BASELINE",
        True,
    )
    REVIEWED_DATA_BASELINE_CSV = Path(
        os.getenv(
            "REVIEWED_DATA_BASELINE_CSV",
            str(RESEARCH_DIR / "data" / "all_image_mask_pairs.csv"),
        )
    )
    REVIEWED_DATA_CLASS_MAPPING = Path(
        os.getenv(
            "REVIEWED_DATA_CLASS_MAPPING",
            str(RESEARCH_DIR / "data" / "class_mapping.json"),
        )
    )

    REVIEWED_DATA_SUPABASE_URL = os.getenv(
        "REVIEWED_DATA_SUPABASE_URL",
        os.getenv("SUPABASE_URL", ""),
    ).rstrip("/")
    REVIEWED_DATA_SUPABASE_SERVICE_ROLE_KEY = os.getenv(
        "REVIEWED_DATA_SUPABASE_SERVICE_ROLE_KEY",
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )
    REVIEWED_DATA_SUPABASE_BUCKET = os.getenv(
        "REVIEWED_DATA_SUPABASE_BUCKET",
        "reviewed-training-data",
    )
    REVIEWED_DATA_SUPABASE_PREFIX = os.getenv(
        "REVIEWED_DATA_SUPABASE_PREFIX",
        "reviewed-data",
    ).strip("/")

    # Snapshot and split-registry paths are runtime state, not source artifacts.
    # They are deliberately kept under deployment/runtime by default.
    RETRAIN_SNAPSHOT_ROOT = Path(
        os.getenv(
            "RETRAIN_SNAPSHOT_ROOT",
            str(DEPLOYMENT_DIR / "runtime" / "retrain_snapshots"),
        )
    )
    RETRAIN_SPLIT_REGISTRY = Path(
        os.getenv(
            "RETRAIN_SPLIT_REGISTRY",
            str(DEPLOYMENT_DIR / "runtime" / "reviewed_split_registry.json"),
        )
    )
    RETRAIN_TFRECORD_SHARDS = max(
        1,
        int(os.getenv("RETRAIN_TFRECORD_SHARDS", "10")),
    )
    RETRAIN_SPLIT_SEED = os.getenv("RETRAIN_SPLIT_SEED", "lung-detection-v1")
    RETRAIN_DRY_RUN = _env_bool("RETRAIN_DRY_RUN", True)
    MODEL_RELEASE_ROOT = Path(
        os.getenv(
            "MODEL_RELEASE_ROOT",
            str(DEPLOYMENT_DIR / "runtime" / "model_releases"),
        )
    )
    ONNX_EXPORT_OPSET = int(os.getenv("ONNX_EXPORT_OPSET", "13"))
    ONNX_VALIDATION_RTOL = float(os.getenv("ONNX_VALIDATION_RTOL", "0.001"))
    ONNX_VALIDATION_ATOL = float(os.getenv("ONNX_VALIDATION_ATOL", "0.001"))

    # Publishing is disabled by default so dry runs can evaluate and stage
    # models without mutating the public Hugging Face artifact repository.
    HF_PUBLISH_ENABLED = _env_bool("HF_PUBLISH_ENABLED", False)
    HF_PUBLISH_REPO_ID = os.getenv(
        "HF_PUBLISH_REPO_ID",
        "a1mohamadd/lung-disease-detection",
    )
    HF_PUBLISH_REPO_TYPE = os.getenv("HF_PUBLISH_REPO_TYPE", "model")
    HF_PUBLISH_REVISION = os.getenv("HF_PUBLISH_REVISION", "main")
    HF_PUBLISH_CREATE_PR = _env_bool("HF_PUBLISH_CREATE_PR", False)
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN", "")
