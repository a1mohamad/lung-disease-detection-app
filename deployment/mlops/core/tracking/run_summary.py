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
        summary["reported_metrics"] = reported_metrics
    if extra:
        summary["extra"] = extra

    return summary


def log_run_summary(summary: dict[str, Any], artifact_path: str = "summary") -> None:
    """Logs a compact JSON artifact that captures run essentials for audit/review."""
    mlflow.log_dict(summary, f"{artifact_path}/run_summary.json")

