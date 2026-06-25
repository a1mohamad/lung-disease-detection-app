from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


ALLOWED_CLASSES = {
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity",
}


def parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ReviewedRecord:
    sample_id: str
    patient_id: str
    image_key: str
    mask_key: str
    class_name: str
    reviewed_at: datetime
    image_sha256: str | None = None
    mask_sha256: str | None = None
    source_id: str = ""


@dataclass(frozen=True)
class ReviewedBatch:
    batch_id: str
    period_start: datetime
    period_end: datetime
    records: tuple[ReviewedRecord, ...]
    source_id: str
    digest: str


def load_reviewed_batch(raw: bytes, *, source_id: str) -> ReviewedBatch:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid reviewed-data manifest: {source_id}") from exc

    if payload.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported manifest schema_version in {source_id}.")

    batch_id = _required_text(payload, "batch_id", source_id)
    period_start = parse_timestamp(
        _required_text(payload, "period_start", source_id),
        "period_start",
    )
    period_end = parse_timestamp(
        _required_text(payload, "period_end", source_id),
        "period_end",
    )
    if period_end <= period_start:
        raise ValueError(f"period_end must be after period_start in {source_id}.")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"records must be a non-empty list in {source_id}.")

    seen_sample_ids: set[str] = set()
    records = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, dict):
            raise ValueError(f"records[{index}] must be an object in {source_id}.")
        record = _load_record(item, source_id=source_id, index=index)
        if record.sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate sample_id '{record.sample_id}' in {source_id}."
            )
        if not period_start <= record.reviewed_at < period_end:
            raise ValueError(
                f"records[{index}].reviewed_at must fall within the batch period "
                f"in {source_id}."
            )
        seen_sample_ids.add(record.sample_id)
        records.append(record)

    return ReviewedBatch(
        batch_id=batch_id,
        period_start=period_start,
        period_end=period_end,
        records=tuple(records),
        source_id=source_id,
        digest=hashlib.sha256(raw).hexdigest(),
    )


def _load_record(
    item: dict[str, Any],
    *,
    source_id: str,
    index: int,
) -> ReviewedRecord:
    prefix = f"records[{index}]"
    class_name = _required_text(item, "class_name", source_id, prefix)
    if class_name not in ALLOWED_CLASSES:
        raise ValueError(
            f"{prefix}.class_name '{class_name}' is not supported in {source_id}."
        )

    image_sha256 = _optional_sha256(item.get("image_sha256"), prefix, source_id)
    mask_sha256 = _optional_sha256(item.get("mask_sha256"), prefix, source_id)

    return ReviewedRecord(
        sample_id=_required_text(item, "sample_id", source_id, prefix),
        patient_id=_required_text(item, "patient_id", source_id, prefix),
        image_key=_safe_key(
            _required_text(item, "image_key", source_id, prefix),
            prefix,
            source_id,
        ),
        mask_key=_safe_key(
            _required_text(item, "mask_key", source_id, prefix),
            prefix,
            source_id,
        ),
        class_name=class_name,
        reviewed_at=parse_timestamp(
            _required_text(item, "reviewed_at", source_id, prefix),
            f"{prefix}.reviewed_at",
        ),
        image_sha256=image_sha256,
        mask_sha256=mask_sha256,
        source_id=source_id,
    )


def _required_text(
    payload: dict[str, Any],
    key: str,
    source_id: str,
    prefix: str = "",
) -> str:
    value = payload.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string in {source_id}.")
    return value.strip()


def _safe_key(value: str, prefix: str, source_id: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized or ".." in normalized.split("/"):
        raise ValueError(f"{prefix} contains an unsafe object key in {source_id}.")
    return normalized


def _optional_sha256(value: Any, prefix: str, source_id: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{prefix} checksum must be text in {source_id}.")
    normalized = value.lower().strip()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{prefix} checksum must be SHA-256 in {source_id}.")
    return normalized
