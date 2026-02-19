from __future__ import annotations

from typing import Optional

import mlflow
from mlflow import MlflowClient


def get_client() -> MlflowClient:
    return MlflowClient()


def get_best_production_metric(model_name: str, metric: str) -> Optional[float]:
    client = get_client()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception:
        return None
    if not versions:
        return None
    run_id = versions[0].run_id
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
        client.transition_model_version_stage(
            name=model_name,
            version=current_version,
            stage="Production",
            archive_existing_versions=True,
        )


def load_model_from_registry(model_name: str, stage: str):
    uri = f"models:/{model_name}/{stage}"
    return mlflow.keras.load_model(uri)
