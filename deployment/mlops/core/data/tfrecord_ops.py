from __future__ import annotations

import math
from pathlib import Path

import tensorflow as tf


def list_tfrecords(tfrecords_dir: Path) -> list[Path]:
    return sorted(tfrecords_dir.glob("*.tfrecord"))


def split_tfrecords(tfrecords: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    if not tfrecords:
        return [], []
    val_count = max(1, int(len(tfrecords) * val_ratio))
    val_files = tfrecords[:val_count]
    train_files = tfrecords[val_count:]
    if not train_files:
        train_files = val_files
    return train_files, val_files


def count_examples_in_tfrecords(tfrecord_paths: list[Path]) -> int:
    total = 0
    for path in tfrecord_paths:
        total += sum(1 for _ in tf.data.TFRecordDataset(str(path)))
    return total


def compute_steps_from_tfrecords(tfrecord_paths: list[Path], batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    examples = count_examples_in_tfrecords(tfrecord_paths)
    if examples == 0:
        return 0
    return math.ceil(examples / batch_size)
