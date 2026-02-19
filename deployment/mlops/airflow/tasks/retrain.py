from __future__ import annotations

from typing import Optional

from mlops.jobs.monthly_retrain import run_pipeline


def retrain_model(
    *,
    model_name: str,
    tfrecords_dir: str,
    batch_size: int = 16,
    epochs: int = 20,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    experiment: str = "lung-detection",
    stage: str = "Production",
    val_ratio: float = 0.2,
) -> None:
    run_pipeline(
        tfrecords_dir=tfrecords_dir,
        batch_size=batch_size,
        epochs=epochs,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        experiment=experiment,
        stage=stage,
        model_name=model_name,
        val_ratio=val_ratio,
    )
