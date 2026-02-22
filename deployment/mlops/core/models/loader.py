from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow

from app.utils.metrics import dice_coefficient
from mlops.core.models.compile import compile_for_task, load_model_local


def load_model_from_registry_or_local(model_name: str, stage: str, local_path: Path, task: str):
    try:
        custom_objects = {"dice_coefficient": dice_coefficient} if task == "segmentation" else None
        return mlflow.keras.load_model(f"models:/{model_name}/{stage}", custom_objects=custom_objects)
    except Exception:
        return load_model_local(str(local_path), task)


def load_compiled_model(spec, stage: str, metadata: dict[str, Any]):
    model_rel = metadata.get("model", {}).get("path", "")
    model_path = spec.model_dir / model_rel
    model = load_model_from_registry_or_local(spec.registered_name, stage, model_path, spec.task)
    return compile_for_task(model, spec.task), model_path
