from __future__ import annotations

from pathlib import Path

from mlflow import MlflowClient


def get_or_create_eval_dataset(
    *,
    client: MlflowClient,
    experiment_id: str,
    spec_name: str,
    stage: str,
    val_ratio: float,
    tfrecords_dir: Path,
) -> str | None:
    dataset_name = f"monthly-eval-{spec_name}"
    try:
        existing = client.search_datasets(
            experiment_ids=[experiment_id],
            filter_string=f"name = '{dataset_name}'",
            max_results=1,
        )
        if existing:
            return existing[0].dataset_id

        dataset = client.create_dataset(
            name=dataset_name,
            experiment_id=experiment_id,
            tags={
                "source": "monthly_log_results",
                "model_name": spec_name,
                "stage": stage,
                "val_ratio": str(val_ratio),
                "tfrecords_dir": str(tfrecords_dir),
            },
        )
        return dataset.dataset_id
    except Exception:
        return None

