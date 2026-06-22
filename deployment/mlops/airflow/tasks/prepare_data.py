from __future__ import annotations

from mlops.config.settings import MLOpsSettings


def prepare_retrain_dataset(**context) -> dict[str, object]:
    mode = MLOpsSettings.RETRAIN_DATASET_MODE
    if mode == "legacy":
        return {
            "snapshot_dir": str(MLOpsSettings.TFRECORDS_DIR),
            "dataset_mode": "legacy",
            "fingerprint": None,
        }
    if mode != "prepared":
        raise ValueError(f"Unsupported RETRAIN_DATASET_MODE: {mode}")

    # Lazy import keeps TensorFlow and Pillow out of Airflow DAG parsing.
    from mlops.core.ingestion.snapshot import prepare_reviewed_snapshot

    interval_end = context.get("data_interval_end") or context.get("logical_date")
    return prepare_reviewed_snapshot(interval_end)
