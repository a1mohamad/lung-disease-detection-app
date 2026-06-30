"""Small MLflow I/O helpers for YAML metadata and parameter flattening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed dictionary, or an empty dictionary for empty YAML files.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested dictionaries into MLflow-compatible string parameters.

    Args:
        data: Nested dictionary to flatten.
        prefix: Dot-prefix used for recursive calls.

    Returns:
        Flat dictionary with string values accepted by MLflow params.
    """
    flat: dict[str, str] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            # MLflow params must be scalar-ish strings; JSON preserves lists in a
            # readable form without losing their exact values.
            if isinstance(value, (list, tuple)):
                flat[new_key] = json.dumps(value)
            else:
                flat[new_key] = str(value)
    return flat
