from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import yaml

from app.utils.metrics import dice_coefficient
from mlops.core.models.compile import compile_for_task, load_model_local


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            if isinstance(value, (list, tuple)):
                flat[new_key] = json.dumps(value)
            else:
                flat[new_key] = str(value)
    return flat


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

