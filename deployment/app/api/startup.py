"""Startup checks and resource factories for the inference API."""

from __future__ import annotations

import numpy as np

from app.configs.config import AppConfig
from app.db.base import Base
from app.db import models  # noqa: F401
from app.db.session import engine
from app.predictor.pipeline import LungDetection
from app.utils.errors import ArtifactError
from app.utils.hf_models import ensure_models_available_from_huggingface
from app.utils.metadata import load_metadata


def check_paths_and_metadata() -> None:
    """Validate required model artifacts before the API accepts traffic.

    The application should fail during startup rather than during the first
    clinical-style prediction request. This check verifies local or downloaded
    model directories, ``metadata.yaml`` files, metadata readability, and the
    prediction artifact output directory.

    Raises:
        ArtifactError: If a required model directory or metadata file is missing.
    """
    ensure_models_available_from_huggingface()

    # Fail fast on artifact problems so broken deployments never become healthy.
    required_dirs = [
        AppConfig.UNET_PATH,
        AppConfig.DENSENET_PATH,
        AppConfig.EFFICIENTNET_PATH,
        AppConfig.INCEPTION_PATH,
        AppConfig.MOBILENET_PATH,
        AppConfig.DISEASE_DENSENET_PATH,
    ]

    for p in required_dirs:
        if not p.exists():
            raise ArtifactError("MODEL_PATH_MISSING", f"Missing model path: {p}")

        meta_path = AppConfig.get_metadata_path(p)
        if not meta_path.exists():
            raise ArtifactError("METADATA_MISSING", f"Missing metadata: {meta_path}")

        _ = load_metadata(p)

    AppConfig.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def create_detector() -> LungDetection:
    """Create the process-wide lung detection pipeline instance.

    Returns:
        Initialized ``LungDetection`` object with all configured model wrappers.
    """
    return LungDetection()


def warmup(detector: LungDetection) -> None:
    """Execute a small prediction to initialize model sessions before traffic.

    Args:
        detector: Process-wide prediction pipeline.

    Notes:
        Warmup catches lazy model/session failures during startup and reduces
        first-request latency after the service becomes healthy.
    """
    # Create a dummy input image (batch 1, configured size, RGB channels). This
    # exercises segmentation and binary prediction without needing real data.
    dummy = np.zeros(
        (1, AppConfig.IMAGE_SIZE[0], AppConfig.IMAGE_SIZE[1], 3), dtype=np.float32
    )
    _ = detector.predict(dummy, return_all=False)


def init_database() -> None:
    """Create prediction logging tables when database logging is enabled.

    The call is intentionally skipped for stateless deployments and cloud
    topologies that disable direct prediction logging.
    """
    if not AppConfig.DB_LOGGING_ENABLED:
        return

    Base.metadata.create_all(bind=engine)
