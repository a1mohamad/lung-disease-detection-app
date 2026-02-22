from __future__ import annotations

import ast
import json
from pathlib import Path


def _flatten_dict(data: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        else:
            flat[new_key] = json.dumps(value) if isinstance(value, (list, tuple)) else str(value)
    return flat


def extract_uppercase_params(notebook_path: Path) -> dict[str, str]:
    if not notebook_path.exists():
        return {}
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    params: dict[str, str] = {}
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        try:
            tree = ast.parse(code)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        value = None
                    params[target.id] = json.dumps(value) if not isinstance(value, str) else value
    return params


def collect_notebook_params(notebooks: list[Path]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for nb in notebooks:
        nb_params = extract_uppercase_params(nb)
        for k, v in nb_params.items():
            merged[f"notebook.{nb.stem}.{k}"] = v
    return merged


def load_optuna_params(paths: list[Path]) -> dict[str, str]:
    params: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in ("best_hparams", "phase1_settings", "phase2_settings"):
            if isinstance(payload.get(key), dict):
                params.update(_flatten_dict(payload[key], f"optuna.{key}"))
        if "best_value" in payload:
            params["optuna.best_value"] = str(payload["best_value"])
        if "best_trial_number" in payload:
            params["optuna.best_trial_number"] = str(payload["best_trial_number"])
    return params

