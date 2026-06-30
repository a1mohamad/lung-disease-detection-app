"""Airflow task wrapper for monthly model evaluation logging."""

from __future__ import annotations

from typing import Optional


def log_model_results(
    *,
    model_name: str,
    tfrecords_dir: str,
    batch_size: int = 16,
    max_eval_batches: Optional[int] = None,
    experiment: str = "lung-detection",
    stage: str = "Production",
    val_ratio: float = 0.2,
) -> None:
    """Run monthly evaluation for one model from an Airflow Python task.

    Args:
        model_name: Managed model short name.
        tfrecords_dir: TFRecord directory to evaluate.
        batch_size: Evaluation batch size.
        max_eval_batches: Optional evaluation cap for smoke tests.
        experiment: MLflow experiment name.
        stage: Model stage/alias to load.
        val_ratio: Legacy validation split ratio.
    """
    # Lazy import prevents heavy ML modules from loading at DAG parse time.
    from mlops.jobs.monthly_log_results import run_pipeline

    run_pipeline(
        tfrecords_dir=tfrecords_dir,
        batch_size=batch_size,
        max_eval_batches=max_eval_batches,
        experiment=experiment,
        stage=stage,
        model_name=model_name,
        val_ratio=val_ratio,
    )
