from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict


def extract_uppercase_params(notebook_path: Path) -> Dict[str, str]:
    if not notebook_path.exists():
        return {}
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    params: Dict[str, str] = {}
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


def collect_notebook_params(notebooks: list[Path]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for nb in notebooks:
        nb_params = extract_uppercase_params(nb)
        for k, v in nb_params.items():
            merged[f"notebook.{nb.stem}.{k}"] = v
    return merged

