from __future__ import annotations

from typing import Optional


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
    register_model: bool = True,
    dataset_mode: str = "legacy",
    **context,
) -> None:
    # Lazy import prevents heavy ML modules from loading at DAG parse time.
    from mlops.jobs.monthly_retrain import run_pipeline

    dag_run = context.get("dag_run")
    dag_conf = getattr(dag_run, "conf", {}) or {}
    run_pipeline(
        tfrecords_dir=dag_conf.get("tfrecords_dir", tfrecords_dir),
        batch_size=batch_size,
        epochs=epochs,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        experiment=experiment,
        stage=stage,
        model_name=model_name,
        val_ratio=val_ratio,
        register_model=register_model,
        dataset_mode=dag_conf.get("dataset_mode", dataset_mode),
    )
