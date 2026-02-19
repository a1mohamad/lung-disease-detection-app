from __future__ import annotations

from pathlib import Path


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

