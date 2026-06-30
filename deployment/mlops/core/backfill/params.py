"""Extract notebook and Optuna metadata for MLflow backfill runs."""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path

NOTEBOOK_PARAM_ALLOWLIST = {
    "AUTO",
    "SEED",
    "BATCH_SIZE",
    "BATCH_SIZE_PER_REPLICA",
    "BATCH_SIZE_PER_REPABLICA",  # keep compatibility with misspelled legacy notebooks
    "IMG_SIZE",
    "IMAGE_SIZE",
    "MASK_SIZE",
    "GLOBAL_BATCH_SIZE",
    "NUM_CLASSES",
    "SHUFFLE_SIZE",
    "INITIAL_EPOCH",
    "MIDTUNE_EPOCH",
    "UNFREEZE_EPOCH",
    "GAIN_EPOCH",
    "WARMUP_EPOCH",
    "FINAL_EPOCH",
    "UNFREEZE_LAYER",
}

BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _is_nullish(value: object) -> bool:
    """Return whether a value should be omitted from logged parameters.

    Args:
        value: Candidate notebook or Optuna parameter value.

    Returns:
        True for empty, null-like, or missing values that would add noise to
        MLflow parameters.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
        return True
    return False


def _flatten_dict(data: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dictionaries into dot-separated string parameters.

    Args:
        data: Nested mapping loaded from an Optuna JSON artifact.
        prefix: Prefix applied to every emitted key.

    Returns:
        Flat string mapping suitable for ``mlflow.log_params``.
    """
    flat: dict[str, str] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        else:
            if _is_nullish(value):
                continue
            flat[new_key] = json.dumps(value) if isinstance(value, (list, tuple)) else str(value)
    return flat


def _try_eval_expr(node: ast.AST, env: dict[str, object]) -> object:
    """Safely evaluate literal-like AST expressions used in notebooks.

    Args:
        node: AST expression node from a notebook code cell.
        env: Previously evaluated simple assignments in the same notebook.

    Returns:
        Evaluated literal-like value, or ``None`` when the expression is too
        dynamic to evaluate safely.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Tuple):
        return tuple(_try_eval_expr(elt, env) for elt in node.elts)
    if isinstance(node, ast.List):
        return [_try_eval_expr(elt, env) for elt in node.elts]
    if isinstance(node, ast.Dict):
        keys = [_try_eval_expr(k, env) for k in node.keys]
        values = [_try_eval_expr(v, env) for v in node.values]
        return dict(zip(keys, values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _try_eval_expr(node.operand, env)
        return -value if isinstance(value, (int, float)) else None
    if isinstance(node, ast.BinOp):
        left = _try_eval_expr(node.left, env)
        right = _try_eval_expr(node.right, env)
        op = BIN_OPS.get(type(node.op))
        if op and isinstance(left, (int, float)) and isinstance(right, (int, float)):
            try:
                return op(left, right)
            except Exception:
                return None
    return None


def _serialize_notebook_value(value: object) -> str:
    """Serialize a notebook parameter value for MLflow logging.

    Args:
        value: Notebook assignment value extracted from a code cell.

    Returns:
        String representation accepted by MLflow parameter logging.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)


def _allow_notebook_key(key: str) -> bool:
    """Return whether an uppercase notebook variable should be logged.

    Args:
        key: Uppercase variable name extracted from a notebook.

    Returns:
        True when the variable looks like a training hyperparameter or known
        research setting worth preserving in MLflow.
    """
    if key.endswith("_DIR"):
        return False
    if key in NOTEBOOK_PARAM_ALLOWLIST:
        return True
    if key.endswith("_LR") or key.endswith("_LRS"):
        return True
    if key.endswith("_EPOCH") or key.endswith("_EPOCHS"):
        return True
    return False


def extract_uppercase_params(notebook_path: Path) -> dict[str, str]:
    """Extract uppercase assignment parameters from notebook code cells.

    Args:
        notebook_path: Notebook file to inspect.

    Returns:
        Mapping of uppercase notebook variable names to serialized values.
    """
    if not notebook_path.exists():
        return {}
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    params: dict[str, str] = {}
    env: dict[str, object] = {}
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        try:
            # Notebook cells can contain exploratory or partially invalid code;
            # skip cells that cannot be parsed instead of failing the backfill.
            tree = ast.parse(code)
        except Exception:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            value = _try_eval_expr(node.value, env)
            if value is None:
                raw = ast.get_source_segment(code, node.value)
                value = raw if raw else None
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = value
                    if target.id.isupper():
                        params[target.id] = _serialize_notebook_value(value)
    return params


def collect_notebook_params(notebooks: list[Path]) -> dict[str, str]:
    """Collect allowed notebook parameters across existing notebooks.

    Args:
        notebooks: Notebook paths associated with one model.

    Returns:
        MLflow-ready parameter dictionary with notebook names and selected
        uppercase configuration values.
    """
    existing = [nb for nb in notebooks if nb.exists()]
    if not existing:
        return {}

    merged: dict[str, str] = {
        "notebook.count": str(len(existing)),
        "notebook.names": json.dumps([nb.stem for nb in existing]),
    }
    for nb in existing:
        nb_params = extract_uppercase_params(nb)
        for key, value in nb_params.items():
            if not _allow_notebook_key(key) or _is_nullish(value):
                continue
            merged[f"notebook.{nb.stem}.{key}"] = value
    return merged


def collect_notebook_support_files(notebooks: list[Path]) -> list[Path]:
    """Find colocated utility files that should be logged with notebooks.

    Args:
        notebooks: Notebook paths associated with one model.

    Returns:
        Deduplicated list of nearby ``utils.py`` files.
    """
    files: list[Path] = []
    seen: set[Path] = set()
    for nb in notebooks:
        if not nb.exists():
            continue
        # include utils.py colocated with the notebook or in ancestor folders (e.g., research/utils.py)
        for parent in [nb.parent, *nb.parents]:
            candidate = parent / "utils.py"
            if candidate.exists() and candidate not in seen:
                seen.add(candidate)
                files.append(candidate)
    return files


def load_optuna_params(paths: list[Path]) -> dict[str, str]:
    """Load selected Optuna result fields from JSON artifacts.

    Args:
        paths: Optuna summary JSON paths.

    Returns:
        MLflow-ready parameter dictionary containing selected Optuna settings
        and best-trial metadata.
    """
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
