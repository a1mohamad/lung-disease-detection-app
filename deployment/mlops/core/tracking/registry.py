"""MLflow registry promotion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow import MlflowClient


def get_client() -> MlflowClient:
    """Return a configured MLflow client.

    Returns:
        MLflow client using the active tracking URI.
    """
    return MlflowClient()


def get_best_production_metric(model_name: str, metric: str) -> Optional[float]:
    """Read the current production alias metric for a registered model.

    Args:
        model_name: MLflow registered model name.
        metric: Metric key to read from the production run.

    Returns:
        Metric value for the production alias, or ``None`` when no production
        alias/metric is available.
    """
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
    """Promotion audit record for one registered model candidate.

    The object captures both the comparison result and whether the alias update
    was actually applied. Logging this structure gives retraining runs an audit
    trail even when promotion is intentionally evaluated in dry-run mode.
    """

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
    """Compare a candidate run against the current production metric.

    Args:
        model_name: MLflow registered model name.
        run_id: Candidate run that produced the newly registered version.
        metric: Metric key used as the promotion criterion.

    Returns:
        Promotion decision containing candidate metric, current production
        metric, registered version, and whether promotion is allowed.
    """
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
    """Apply the MLflow production alias when a decision allows promotion.

    The function is intentionally a no-op for incomplete or rejected decisions,
    making it safe to call from orchestration code after every comparison.
    """
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
    """Return or apply a promotion decision based on candidate metric quality.

    Args:
        model_name: MLflow registered model name.
        run_id: Candidate run id.
        metric: Metric to maximize.
        allow_promotion: When false, compute the decision without mutating the
            registry alias.

    Returns:
        Promotion decision, with ``promoted=True`` only when the alias was
        updated.
    """
    decision = get_promotion_decision(model_name, run_id, metric)
    return apply_promotion(decision) if allow_promotion else decision


def load_model_from_registry(model_name: str, stage: str):
    """Load a Keras model from MLflow using alias-first URI resolution.

    Args:
        model_name: MLflow registered model name.
        stage: Alias or legacy stage name to load.

    Returns:
        Loaded Keras model.
    """
    # Prefer aliases (`models:/name@production`) and fallback to legacy stage URIs.
    alias = stage.strip().lower()
    alias_uri = f"models:/{model_name}@{alias}"
    try:
        return mlflow.keras.load_model(alias_uri)
    except Exception:
        uri = f"models:/{model_name}/{stage}"
        return mlflow.keras.load_model(uri)
