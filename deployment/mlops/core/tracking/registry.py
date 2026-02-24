from __future__ import annotations

from typing import Optional

import mlflow
from mlflow import MlflowClient


def get_client() -> MlflowClient:
    return MlflowClient()


def get_best_production_metric(model_name: str, metric: str) -> Optional[float]:
    client = get_client()
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
    except Exception:
        return None
    run_id = prod_version.run_id
    run = client.get_run(run_id)
    return run.data.metrics.get(metric)


def promote_if_better(model_name: str, run_id: str, metric: str) -> None:
    client = get_client()
    best_value = get_best_production_metric(model_name, metric)

    versions = client.search_model_versions(f"name='{model_name}'")
    current_version = None
    for v in versions:
        if v.run_id == run_id:
            current_version = v.version
            break

    if current_version is None:
        return

    run = client.get_run(run_id)
    candidate = run.data.metrics.get(metric)

    if best_value is None or (candidate is not None and candidate > best_value):
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=current_version,
        )


def load_model_from_registry(model_name: str, stage: str):
    # Prefer aliases (`models:/name@production`) and fallback to legacy stage URIs.
    alias = stage.strip().lower()
    alias_uri = f"models:/{model_name}@{alias}"
    try:
        return mlflow.keras.load_model(alias_uri)
    except Exception:
        uri = f"models:/{model_name}/{stage}"
        return mlflow.keras.load_model(uri)
