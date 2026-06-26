from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class PromotionDecision:
    model_name: str
    run_id: str
    version: str | None
    metric: str
    candidate_value: float | None
    production_value: float | None
    should_promote: bool
    promoted: bool = False


def get_promotion_decision(
    model_name: str,
    run_id: str,
    metric: str,
) -> PromotionDecision:
    client = get_client()
    best_value = get_best_production_metric(model_name, metric)

    versions = client.search_model_versions(f"name='{model_name}'")
    current_version = None
    for v in versions:
        if v.run_id == run_id:
            current_version = v.version
            break

    if current_version is None:
        return PromotionDecision(
            model_name=model_name,
            run_id=run_id,
            version=None,
            metric=metric,
            candidate_value=None,
            production_value=best_value,
            should_promote=False,
        )

    run = client.get_run(run_id)
    candidate = run.data.metrics.get(metric)
    should_promote = best_value is None or (
        candidate is not None and candidate > best_value
    )
    return PromotionDecision(
        model_name=model_name,
        run_id=run_id,
        version=current_version,
        metric=metric,
        candidate_value=candidate,
        production_value=best_value,
        should_promote=should_promote,
    )


def apply_promotion(decision: PromotionDecision) -> PromotionDecision:
    if not decision.should_promote or decision.version is None:
        return decision
    get_client().set_registered_model_alias(
        name=decision.model_name,
        alias="production",
        version=decision.version,
    )
    return PromotionDecision(**{**decision.__dict__, "promoted": True})


def promote_if_better(
    model_name: str,
    run_id: str,
    metric: str,
    *,
    allow_promotion: bool = True,
) -> PromotionDecision:
    decision = get_promotion_decision(model_name, run_id, metric)
    return apply_promotion(decision) if allow_promotion else decision


def load_model_from_registry(model_name: str, stage: str):
    # Prefer aliases (`models:/name@production`) and fallback to legacy stage URIs.
    alias = stage.strip().lower()
    alias_uri = f"models:/{model_name}@{alias}"
    try:
        return mlflow.keras.load_model(alias_uri)
    except Exception:
        uri = f"models:/{model_name}/{stage}"
        return mlflow.keras.load_model(uri)
