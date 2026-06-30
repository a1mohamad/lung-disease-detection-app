"""Run summary construction and MLflow artifact logging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import mlflow


def build_run_summary(
    *,
    run_type: str,
    model_name: str,
    task: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    reported_metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact JSON-serializable summary for an MLOps run.

    Args:
        run_type: Logical run type such as evaluation or retraining.
        model_name: Managed model short name.
        task: Model task family.
        metrics: Metrics logged for the run.
        params: Run parameters logged to MLflow.
        reported_metrics: Optional metrics copied from deployment metadata.
        extra: Optional workflow-specific audit fields.

    Returns:
        JSON-serializable run summary dictionary.
    """
    # Keep the artifact compact and predictable so reviewers can compare runs
    # without opening the full MLflow UI.
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": run_type,
        "model": {
            "name": model_name,
            "task": task,
        },
        "metrics": metrics,
        "params": params,
    }

    if reported_metrics:
        # Reported metrics come from deployment metadata and provide historical
        # context next to freshly computed metrics.
        summary["reported_metrics"] = reported_metrics
    if extra:
        # Extra fields capture workflow-specific audit details such as release
        # paths, dataset ids, or publish status.
        summary["extra"] = extra

    return summary


def log_run_summary(summary: dict[str, Any], artifact_path: str = "summary") -> None:
    """Log a compact JSON artifact that captures run essentials for audit.

    Args:
        summary: Summary dictionary produced by ``build_run_summary``.
        artifact_path: MLflow artifact directory for the summary.
    """
    # Logging a single JSON file makes it easy to download or diff run metadata
    # from MLflow artifact storage.
    mlflow.log_dict(summary, f"{artifact_path}/run_summary.json")
