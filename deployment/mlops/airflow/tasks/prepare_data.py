"""Airflow task wrapper for retraining dataset preparation."""

from __future__ import annotations

from mlops.config.settings import MLOpsSettings


def prepare_retrain_dataset(**context) -> dict[str, object]:
    """Resolve or build the TFRecord dataset used by downstream retraining DAGs.

    Args:
        **context: Airflow task context. ``data_interval_end`` or
            ``logical_date`` is used as the reviewed-data snapshot cutoff.

    Returns:
        Dataset descriptor containing snapshot directory, dataset mode, and
        optional fingerprint.

    Raises:
        ValueError: If ``RETRAIN_DATASET_MODE`` is unsupported.
    """
    mode = MLOpsSettings.RETRAIN_DATASET_MODE
    if mode == "legacy":
        return {
            "snapshot_dir": str(MLOpsSettings.TFRECORDS_DIR),
            "dataset_mode": "legacy",
            "fingerprint": None,
        }
    if mode != "prepared":
        raise ValueError(f"Unsupported RETRAIN_DATASET_MODE: {mode}")

    # Lazy import keeps TensorFlow and Pillow out of Airflow DAG parsing, which
    # makes the scheduler faster and less fragile.
    from mlops.core.ingestion.snapshot import prepare_reviewed_snapshot

    interval_end = context.get("data_interval_end") or context.get("logical_date")
    return prepare_reviewed_snapshot(interval_end)
