"""TFRecord listing, splitting, and counting helpers."""

from __future__ import annotations

import math
from pathlib import Path

import tensorflow as tf


def list_tfrecords(tfrecords_dir: Path) -> list[Path]:
    """Return sorted TFRecord files from a directory.

    Args:
        tfrecords_dir: Directory to scan for ``*.tfrecord`` files.

    Returns:
        Deterministically ordered list of TFRecord paths.
    """
    return sorted(tfrecords_dir.glob("*.tfrecord"))


def split_tfrecords(tfrecords: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    """Split flat TFRecord files into train and validation subsets.

    Args:
        tfrecords: Flat list of legacy TFRecord shard paths.
        val_ratio: Fraction of shards to reserve for validation.

    Returns:
        Tuple of ``(train_files, validation_files)``.
    """
    if not tfrecords:
        return [], []
    # Legacy datasets do not have explicit split folders, so split by sorted
    # shard order while guaranteeing at least one validation shard.
    val_count = max(1, int(len(tfrecords) * val_ratio))
    val_files = tfrecords[:val_count]
    train_files = tfrecords[val_count:]
    if not train_files:
        train_files = val_files
    return train_files, val_files


def resolve_tfrecord_splits(
    tfrecords_dir: Path,
    *,
    dataset_mode: str,
    val_ratio: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Resolve train, validation, and optional test files for a dataset mode.

    Args:
        tfrecords_dir: Root directory containing either flat legacy TFRecords or
            prepared ``train``/``validation``/``test`` folders.
        dataset_mode: ``legacy`` or ``prepared``.
        val_ratio: Validation split ratio used only for legacy flat datasets.

    Returns:
        Tuple of ``(train_files, validation_files, test_files)``. Legacy mode
        returns an empty test list.

    Raises:
        ValueError: If the dataset mode is unknown.
        RuntimeError: If a prepared snapshot is missing required split folders.
    """
    mode = dataset_mode.strip().lower()
    if mode == "legacy":
        train_files, val_files = split_tfrecords(
            list_tfrecords(tfrecords_dir),
            val_ratio,
        )
        return train_files, val_files, []
    if mode != "prepared":
        raise ValueError(f"Unsupported retraining dataset mode: {dataset_mode}")

    train_files = list_tfrecords(tfrecords_dir / "train")
    val_files = list_tfrecords(tfrecords_dir / "validation")
    test_files = list_tfrecords(tfrecords_dir / "test")
    if not train_files or not val_files or not test_files:
        raise RuntimeError(
            "Prepared TFRecord snapshot must contain train, validation, and test folders."
        )
    return train_files, val_files, test_files


def count_examples_in_tfrecords(tfrecord_paths: list[Path]) -> int:
    """Count serialized examples across TFRecord files.

    Args:
        tfrecord_paths: TFRecord files to scan.

    Returns:
        Total number of serialized examples.
    """
    total = 0
    for path in tfrecord_paths:
        total += sum(1 for _ in tf.data.TFRecordDataset(str(path)))
    return total


def compute_steps_from_tfrecords(tfrecord_paths: list[Path], batch_size: int) -> int:
    """Compute the number of batches represented by TFRecord files.

    Args:
        tfrecord_paths: TFRecord files to count.
        batch_size: Batch size used by the training or evaluation job.

    Returns:
        Number of batches required to cover all examples.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    examples = count_examples_in_tfrecords(tfrecord_paths)
    if examples == 0:
        return 0
    return math.ceil(examples / batch_size)
