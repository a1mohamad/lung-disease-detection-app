from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from keras.models import load_model

from app.configs.config import AppConfig
from app.utils.errors import ModelError

logger = logging.getLogger(__name__)


def _load_from_registry(model_name: str, custom_objects: Optional[dict[str, Any]] = None):
    import mlflow

    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
    uri = f"models:/{model_name}/{AppConfig.MLFLOW_MODEL_STAGE}"
    return mlflow.keras.load_model(uri, custom_objects=custom_objects)


def load_keras_model(
    *,
    model_dir: Path,
    model_rel_path: str,
    model_name: str,
    custom_objects: Optional[dict[str, Any]] = None,
):
    local_model_path = str(model_dir / model_rel_path)
    model_uri = f"models:/{model_name}/{AppConfig.MLFLOW_MODEL_STAGE}"

    if AppConfig.MLFLOW_ENABLED:
        try:
            model = _load_from_registry(model_name, custom_objects=custom_objects)
            logger.info(
                "Model loaded from MLflow registry. uri=%s tracking_uri=%s",
                model_uri,
                AppConfig.MLFLOW_TRACKING_URI,
            )
            return model
        except Exception as exc:
            details = {
                "model_name": model_name,
                "model_stage": AppConfig.MLFLOW_MODEL_STAGE,
                "tracking_uri": AppConfig.MLFLOW_TRACKING_URI,
                "model_uri": model_uri,
                "local_fallback_path": local_model_path,
                "error": str(exc),
            }
            logger.warning(
                "MLflow model load failed. Falling back to local model. details=%s",
                details,
            )
            if AppConfig.MLFLOW_STRICT:
                raise ModelError(
                    "MLFLOW_MODEL_LOAD_FAILED",
                    "Failed to load model from MLflow registry and strict mode is enabled.",
                    details,
                ) from exc

    model = load_model(local_model_path, custom_objects=custom_objects)
    logger.info("Model loaded from local path. path=%s", local_model_path)
    return model
