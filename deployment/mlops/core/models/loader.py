"""Model loading helpers for MLflow registry fallback to local artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from app.utils.metrics import dice_coefficient
from mlops.core.models.compile import compile_for_task, load_model_local


def load_model_from_registry_or_local(model_name: str, stage: str, local_path: Path, task: str):
    """Load a model from MLflow when available, otherwise from a local path.

    Args:
        model_name: MLflow registered model name.
        stage: Registry stage or alias to load.
        local_path: Local Keras artifact path used as fallback.
        task: Model task, used to attach custom segmentation objects.

    Returns:
        Loaded Keras model.
    """
    try:
        custom_objects = {"dice_coefficient": dice_coefficient} if task == "segmentation" else None
        return mlflow.keras.load_model(f"models:/{model_name}/{stage}", custom_objects=custom_objects)
    except Exception:
        # Local fallback keeps evaluation and retraining usable before the
        # registry has been bootstrapped.
        return load_model_local(str(local_path), task)


def load_compiled_model(spec, stage: str, metadata: dict[str, Any]):
    """Load and compile the model described by a model spec and metadata.

    Args:
        spec: Model specification containing local path, task, and registry name.
        stage: MLflow stage or alias to try first.
        metadata: Parsed metadata containing the relative model path.

    Returns:
        Tuple of ``(compiled_model, local_model_path)``.
    """
    model_rel = metadata.get("model", {}).get("path", "")
    model_path = spec.model_dir / model_rel
    model = load_model_from_registry_or_local(spec.registered_name, stage, model_path, spec.task)
    return compile_for_task(model, spec.task), model_path
