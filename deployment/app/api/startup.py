from __future__ import annotations

import tensorflow as tf

from app.configs.config import AppConfig
from app.db.base import Base
from app.db import models  # noqa: F401
from app.db.session import engine
from app.predictor.pipeline import LungDetection
from app.utils.errors import ArtifactError
from app.utils.metadata import load_metadata


def check_paths_and_metadata() -> None:
    # model directories
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

        # validate metadata.yaml exists and is readable
        meta_path = AppConfig.get_metadata_path(p)
        if not meta_path.exists():
            raise ArtifactError("METADATA_MISSING", f"Missing metadata: {meta_path}")

        # will raise if invalid
        _ = load_metadata(p)

    # output folder
    AppConfig.PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def create_detector() -> LungDetection:
    return LungDetection()


def warmup(detector: LungDetection) -> None:
    # create a dummy input image (batch 1, 256x256, 3)
    dummy = tf.zeros(
        (1, AppConfig.IMAGE_SIZE[0], AppConfig.IMAGE_SIZE[1], 3), dtype=tf.float32
    )
    _ = detector.predict(dummy, return_all=False)


def init_database() -> None:
    if not AppConfig.DB_LOGGING_ENABLED:
        return

    Base.metadata.create_all(bind=engine)
