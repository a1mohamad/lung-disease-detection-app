"""Reviewed-data snapshot builder for prepared retraining datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable

from mlops.config.settings import MLOpsSettings
from mlops.core.ingestion.manifest import ReviewedBatch, ReviewedRecord, load_reviewed_batch
from mlops.core.ingestion.splits import SPLITS, SplitRegistry
from mlops.core.ingestion.storage import (
    LocalReviewedDataStorage,
    ReviewedDataStorage,
    build_reviewed_data_storage,
)


@dataclass(frozen=True)
class SnapshotRecord:
    """Reviewed record plus lazy image and mask readers.

    Attributes:
        record: Reviewed-data manifest record.
        read_image: Callable that loads raw image bytes on demand.
        read_mask: Callable that loads raw mask bytes on demand.
    """

    record: ReviewedRecord
    read_image: Callable[[], bytes]
    read_mask: Callable[[], bytes]


def prepare_reviewed_snapshot(
    interval_end: datetime | str | None = None,
) -> dict[str, object]:
    """Build or reuse a fingerprinted TFRecord snapshot for reviewed data.

    The snapshot is the handoff point between reviewed clinical-style examples
    and retraining jobs. It gathers eligible reviewed batches, optionally mixes
    in the baseline research dataset, applies stable patient-level splits,
    writes TFRecord shards, and records a manifest that can be audited later.

    Args:
        interval_end: Inclusive cutoff timestamp for reviewed batches. Strings
            are parsed as ISO-8601 and normalized to UTC.

    Returns:
        Snapshot metadata including directory, dataset mode, fingerprint, and
        per-split counts.

    Raises:
        RuntimeError: If no reviewed data is available, a conflicting snapshot
        exists, or TFRecord generation cannot complete.
    """
    cutoff = _parse_interval_end(interval_end)
    class_mapping = _load_class_mapping(MLOpsSettings.REVIEWED_DATA_CLASS_MAPPING)
    records, sources = _collect_records(cutoff)
    if not records:
        raise RuntimeError("No reviewed records are available for this retraining snapshot.")

    # The fingerprint makes snapshot creation idempotent: the same cutoff,
    # sources, classes, split seed, and shard count resolve to the same dataset.
    fingerprint = _snapshot_fingerprint(cutoff, sources, class_mapping)
    snapshot_id = cutoff.strftime("snapshot-%Y%m%dT%H%M%SZ")
    snapshot_dir = MLOpsSettings.RETRAIN_SNAPSHOT_ROOT / snapshot_id
    existing = _existing_snapshot(snapshot_dir, fingerprint)
    if existing is not None:
        return existing

    registry = SplitRegistry.load(
        MLOpsSettings.RETRAIN_SPLIT_REGISTRY,
        MLOpsSettings.RETRAIN_SPLIT_SEED,
    )
    assignments = registry.assign(item.record for item in records)

    # Write to a hidden temporary directory first so downstream jobs never read
    # a half-written snapshot as if it were production-ready.
    temp_dir = snapshot_dir.with_name(f".{snapshot_id}.tmp")
    if temp_dir.exists():
        raise RuntimeError(
            f"Temporary snapshot directory already exists: {temp_dir}. "
            "Inspect and remove it before retrying."
        )
    temp_dir.mkdir(parents=True)

    try:
        counts = _write_snapshot(
            records=records,
            assignments=assignments,
            class_mapping=class_mapping,
            output_dir=temp_dir,
        )
        metadata = {
            "schema_version": "1.0",
            "snapshot_id": snapshot_id,
            "fingerprint": fingerprint,
            "interval_end": cutoff.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_mode": "prepared",
            "counts": dict(counts),
            "class_mapping": class_mapping,
            "sources": sources,
            "split_registry": str(MLOpsSettings.RETRAIN_SPLIT_REGISTRY),
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        registry.save()
        temp_dir.replace(snapshot_dir)
    except Exception:
        # Keep the temporary directory for inspection; never present it as complete.
        raise

    return {
        "snapshot_dir": str(snapshot_dir),
        "dataset_mode": "prepared",
        "fingerprint": fingerprint,
        "counts": dict(counts),
    }


def _collect_records(
    cutoff: datetime,
) -> tuple[list[SnapshotRecord], list[dict[str, object]]]:
    """Collect baseline and reviewed records available before the cutoff.

    Args:
        cutoff: Inclusive snapshot cutoff normalized to UTC.

    Returns:
        Snapshot records plus source metadata used for fingerprinting.

    Raises:
        ValueError: If duplicate sample ids are found across sources.
    """
    records: list[SnapshotRecord] = []
    sources: list[dict[str, object]] = []

    if MLOpsSettings.REVIEWED_DATA_INCLUDE_BASELINE:
        baseline_records, baseline_source = _load_baseline_records()
        records.extend(baseline_records)
        sources.append(baseline_source)

    storage = build_reviewed_data_storage()
    for manifest_key in storage.list_manifest_keys():
        raw = storage.read_bytes(manifest_key)
        batch = load_reviewed_batch(raw, source_id=manifest_key)
        if batch.period_end > cutoff:
            continue
        included = _records_from_batch(batch, storage, cutoff)
        records.extend(included)
        sources.append(
            {
                "type": MLOpsSettings.REVIEWED_DATA_BACKEND,
                "manifest_key": manifest_key,
                "batch_id": batch.batch_id,
                "digest": batch.digest,
                "record_count": len(included),
            }
        )

    seen: set[str] = set()
    for item in records:
        sample_id = item.record.sample_id
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id across reviewed sources: {sample_id}")
        seen.add(sample_id)
    return records, sources


def _load_baseline_records() -> tuple[list[SnapshotRecord], dict[str, object]]:
    """Load legacy research data as baseline records for reviewed snapshots.

    Returns:
        Baseline snapshot records and a source descriptor for the CSV input.

    Raises:
        FileNotFoundError: If the configured baseline CSV is missing.
        ValueError: If the CSV lacks required columns.
    """
    csv_path = MLOpsSettings.REVIEWED_DATA_BASELINE_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"Baseline training CSV not found: {csv_path}")

    storage = LocalReviewedDataStorage(
        MLOpsSettings.RESEARCH_DIR,
        manifest_index="unused.json",
    )
    records: list[SnapshotRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "mask_path", "class"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Baseline CSV is missing required columns: {csv_path}")
        for row_number, row in enumerate(reader, start=2):
            image_key = _baseline_key(row["image_path"])
            mask_key = _baseline_key(row["mask_path"])
            class_name = row["class"].strip()
            identity = hashlib.sha256(image_key.encode("utf-8")).hexdigest()[:20]
            record = ReviewedRecord(
                sample_id=f"baseline-{identity}",
                patient_id=f"baseline-{identity}",
                image_key=image_key,
                mask_key=mask_key,
                class_name=class_name,
                reviewed_at=datetime(1970, 1, 1, tzinfo=timezone.utc),
                source_id=f"{csv_path}:{row_number}",
            )
            records.append(
                SnapshotRecord(
                    record=record,
                    read_image=lambda key=image_key: storage.read_bytes(key),
                    read_mask=lambda key=mask_key: storage.read_bytes(key),
                )
            )

    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    return records, {
        "type": "baseline_csv",
        "path": str(csv_path),
        "digest": digest,
        "record_count": len(records),
    }


def _records_from_batch(
    batch: ReviewedBatch,
    storage: ReviewedDataStorage,
    cutoff: datetime,
) -> list[SnapshotRecord]:
    """Convert a reviewed manifest batch into snapshot records.

    Args:
        batch: Parsed reviewed-data manifest batch.
        storage: Storage adapter used to lazily read image and mask bytes.
        cutoff: Inclusive cutoff; records at or after this time are skipped.

    Returns:
        Snapshot records with lazy image and mask readers.
    """
    result = []
    for original in batch.records:
        if original.reviewed_at >= cutoff:
            continue
        record = replace(original, source_id=batch.source_id)
        result.append(
            SnapshotRecord(
                record=record,
                read_image=lambda key=record.image_key: storage.read_bytes(key),
                read_mask=lambda key=record.mask_key: storage.read_bytes(key),
            )
        )
    return result


def _write_snapshot(
    *,
    records: list[SnapshotRecord],
    assignments: dict[str, str],
    class_mapping: dict[str, int],
    output_dir: Path,
) -> Counter:
    """Write split TFRecord shards and a JSONL inventory for a snapshot.

    Each record is checksum-validated, converted to canonical PNG bytes, routed
    to the patient-level split assigned by ``SplitRegistry``, and written to a
    deterministic shard based on ``sample_id``.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to create TFRecord snapshots.") from exc

    shard_count = MLOpsSettings.RETRAIN_TFRECORD_SHARDS
    writers: dict[tuple[str, int], object] = {}
    counts: Counter = Counter()
    inventory_path = output_dir / "records.jsonl"

    for split in SPLITS:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True)
        for shard in range(shard_count):
            path = split_dir / f"part-{shard:05d}-of-{shard_count:05d}.tfrecord"
            writers[(split, shard)] = tf.io.TFRecordWriter(str(path))

    try:
        with inventory_path.open("w", encoding="utf-8", newline="\n") as inventory:
            for item in sorted(records, key=lambda value: value.record.sample_id):
                record = item.record
                if record.class_name not in class_mapping:
                    raise ValueError(
                        f"Unknown class '{record.class_name}' for {record.sample_id}."
                    )
                split = assignments[record.patient_id]
                image_raw = item.read_image()
                mask_raw = item.read_mask()
                _check_checksum(image_raw, record.image_sha256, record, "image")
                _check_checksum(mask_raw, record.mask_sha256, record, "mask")
                image_png = _canonical_png(image_raw, "RGB", record, "image")
                mask_png = _canonical_png(mask_raw, "L", record, "mask")

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": _bytes_feature(image_png, tf),
                            "mask": _bytes_feature(mask_png, tf),
                            "class": _int64_feature(class_mapping[record.class_name], tf),
                        }
                    )
                )
                shard = int(
                    hashlib.sha256(record.sample_id.encode("utf-8")).hexdigest(),
                    16,
                ) % shard_count
                writers[(split, shard)].write(example.SerializeToString())
                counts[split] += 1
                inventory.write(
                    json.dumps(
                        {
                            "sample_id": record.sample_id,
                            "patient_id": record.patient_id,
                            "class_name": record.class_name,
                            "split": split,
                            "source_id": record.source_id,
                            "image_key": record.image_key,
                            "mask_key": record.mask_key,
                            "image_sha256": hashlib.sha256(image_raw).hexdigest(),
                            "mask_sha256": hashlib.sha256(mask_raw).hexdigest(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        for writer in writers.values():
            writer.close()

    empty_splits = [split for split in SPLITS if counts[split] == 0]
    if empty_splits:
        raise RuntimeError(
            "Prepared snapshot has empty split(s): "
            f"{', '.join(empty_splits)}. Add more reviewed patients before retraining."
        )
    return counts


def _canonical_png(
    raw: bytes,
    mode: str,
    record: ReviewedRecord,
    kind: str,
) -> bytes:
    """Validate image bytes and encode them as canonical PNG.

    Args:
        raw: Original image or mask bytes from storage.
        mode: Pillow conversion mode such as ``RGB`` or ``L``.
        record: Reviewed record used for contextual error messages.
        kind: Human-readable object kind, usually ``image`` or ``mask``.

    Returns:
        PNG-encoded bytes in the requested mode.

    Raises:
        RuntimeError: If Pillow is not installed.
        ValueError: If the object cannot be decoded as an image.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to validate reviewed images.") from exc

    try:
        with Image.open(BytesIO(raw)) as image:
            converted = image.convert(mode)
            output = BytesIO()
            converted.save(output, format="PNG")
            return output.getvalue()
    except Exception as exc:
        raise ValueError(
            f"Invalid {kind} for sample '{record.sample_id}' ({record.source_id})."
        ) from exc


def _check_checksum(
    raw: bytes,
    expected: str | None,
    record: ReviewedRecord,
    kind: str,
) -> None:
    """Validate an optional SHA-256 checksum for a reviewed object.

    Args:
        raw: Object bytes to validate.
        expected: Optional expected SHA-256 hex digest.
        record: Reviewed record used for contextual error messages.
        kind: Human-readable object kind, usually ``image`` or ``mask``.

    Raises:
        ValueError: If a checksum is present and does not match.
    """
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(
            f"{kind} checksum mismatch for sample '{record.sample_id}' "
            f"({record.source_id})."
        )


def _bytes_feature(value: bytes, tf):
    """Create a bytes feature for TFRecord serialization.

    Args:
        value: Raw bytes to store.
        tf: TensorFlow module imported lazily by the caller.

    Returns:
        TensorFlow bytes feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int, tf):
    """Create an int64 feature for TFRecord serialization.

    Args:
        value: Integer class id to store.
        tf: TensorFlow module imported lazily by the caller.

    Returns:
        TensorFlow int64 feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _load_class_mapping(path: Path) -> dict[str, int]:
    """Load and validate the expected four-class mapping.

    Args:
        path: JSON mapping from class names to integer ids.

    Returns:
        Validated mapping for the supported lung-disease classes.

    Raises:
        FileNotFoundError: If the mapping file is missing.
        ValueError: If the mapping schema or label set is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid class mapping: {path}")
    mapping = {str(key): int(value) for key, value in payload.items()}
    if set(mapping) != {"COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"}:
        raise ValueError(f"Class mapping does not match the supported labels: {path}")
    return mapping


def _snapshot_fingerprint(
    cutoff: datetime,
    sources: list[dict[str, object]],
    class_mapping: dict[str, int],
) -> str:
    """Compute a deterministic snapshot fingerprint from inputs and settings.

    Args:
        cutoff: Snapshot cutoff timestamp.
        sources: Source descriptors included in the snapshot.
        class_mapping: Class-name to integer-id mapping.

    Returns:
        SHA-256 digest representing the snapshot inputs and key settings.
    """
    payload = {
        "interval_end": cutoff.isoformat(),
        "sources": sources,
        "class_mapping": class_mapping,
        "split_seed": MLOpsSettings.RETRAIN_SPLIT_SEED,
        "shards": MLOpsSettings.RETRAIN_TFRECORD_SHARDS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_snapshot(
    snapshot_dir: Path,
    fingerprint: str,
) -> dict[str, object] | None:
    """Return an existing snapshot when its fingerprint matches.

    Args:
        snapshot_dir: Expected snapshot output directory.
        fingerprint: Fingerprint for the requested snapshot inputs.

    Returns:
        Existing snapshot metadata when reusable, otherwise ``None``.

    Raises:
        RuntimeError: If the directory exists but is incomplete or mismatched.
    """
    if not snapshot_dir.exists():
        return None
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Incomplete snapshot directory already exists: {snapshot_dir}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise RuntimeError(
            f"Snapshot path already exists with different contents: {snapshot_dir}"
        )
    return {
        "snapshot_dir": str(snapshot_dir),
        "dataset_mode": "prepared",
        "fingerprint": fingerprint,
        "counts": payload.get("counts", {}),
    }


def _baseline_key(value: str) -> str:
    """Normalize a baseline CSV path into a safe storage key.

    Args:
        value: Path value read from the baseline CSV.

    Returns:
        Clean root-relative key usable by the local storage adapter.

    Raises:
        ValueError: If the path is empty or attempts traversal.
    """
    clean = value.strip().replace("\\", "/")
    while clean.startswith("./"):
        clean = clean[2:]
    if not clean or ".." in clean.split("/"):
        raise ValueError(f"Unsafe baseline path: {value}")
    return clean


def _parse_interval_end(value: datetime | str | None) -> datetime:
    """Parse a snapshot cutoff timestamp and normalize it to UTC.

    Args:
        value: Datetime, ISO-8601 string, or ``None`` for current UTC time.

    Returns:
        Timezone-aware UTC datetime.

    Raises:
        ValueError: If a string cannot be parsed as ISO-8601.
    """
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("interval_end must be an ISO-8601 timestamp.") from exc
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
