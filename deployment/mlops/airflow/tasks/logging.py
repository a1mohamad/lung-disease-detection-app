from __future__ import annotations

from typing import Optional

from mlops.jobs.monthly_log_results import run_pipeline


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
    run_pipeline(
        tfrecords_dir=tfrecords_dir,
        batch_size=batch_size,
        max_eval_batches=max_eval_batches,
        experiment=experiment,
        stage=stage,
        model_name=model_name,
        val_ratio=val_ratio,
    )
