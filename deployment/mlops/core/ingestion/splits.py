"""Stable patient-level split assignment for reviewed-data snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mlops.core.ingestion.manifest import ReviewedRecord


SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}


@dataclass
class SplitRegistry:
    """Persistent patient-to-split assignment registry.

    The registry prevents data leakage by assigning whole patients, not single
    images, to train/validation/test splits. Assignments are saved so future
    reviewed snapshots keep the same patient in the same split.
    """

    path: Path
    seed: str
    assignments: dict[str, dict]

    @classmethod
    def load(cls, path: Path, seed: str) -> "SplitRegistry":
        """Load an existing split registry or create an empty one.

        Args:
            path: Registry JSON path.
            seed: Split seed expected by this project configuration.

        Returns:
            Loaded or empty split registry.

        Raises:
            ValueError: If schema, seed, or assignments are invalid.
        """
        if not path.exists():
            return cls(path=path, seed=seed, assignments={})
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise ValueError(f"Unsupported split registry schema: {path}")
        stored_seed = payload.get("seed")
        if stored_seed != seed:
            raise ValueError(
                f"Split registry seed mismatch: expected '{seed}', found '{stored_seed}'."
            )
        assignments = payload.get("assignments")
        if not isinstance(assignments, dict):
            raise ValueError(f"Invalid split registry assignments: {path}")
        return cls(path=path, seed=seed, assignments=assignments)

    def assign(self, records: Iterable[ReviewedRecord]) -> dict[str, str]:
        """Assign patients to train, validation, or test splits.

        Args:
            records: Reviewed records to include in the next snapshot.

        Returns:
            Mapping from ``patient_id`` to split name.

        Raises:
            ValueError: If a patient has conflicting labels, changes labels
            across snapshots, or has an invalid saved split.
        """
        patients: dict[str, list[ReviewedRecord]] = defaultdict(list)
        for record in records:
            patients[record.patient_id].append(record)

        patient_labels: dict[str, str] = {}
        for patient_id, patient_records in patients.items():
            labels = {record.class_name for record in patient_records}
            if len(labels) != 1:
                raise ValueError(
                    f"Patient '{patient_id}' has conflicting class labels: {sorted(labels)}"
                )
            patient_labels[patient_id] = next(iter(labels))

        result: dict[str, str] = {}
        new_by_label: dict[str, list[str]] = defaultdict(list)
        # Existing assignments are immutable by design; moving a patient after
        # previous evaluation would contaminate longitudinal model comparisons.
        for patient_id, label in patient_labels.items():
            existing = self.assignments.get(patient_id)
            if existing:
                if existing.get("class_name") != label:
                    raise ValueError(
                        f"Patient '{patient_id}' changed class label from "
                        f"'{existing.get('class_name')}' to '{label}'."
                    )
                split = existing.get("split")
                if split not in SPLITS:
                    raise ValueError(f"Patient '{patient_id}' has invalid saved split.")
                result[patient_id] = split
            else:
                new_by_label[label].append(patient_id)

        for label, new_patient_ids in sorted(new_by_label.items()):
            current_counts = Counter(
                item["split"]
                for item in self.assignments.values()
                if item.get("class_name") == label and item.get("split") in SPLITS
            )
            total_after = sum(current_counts.values()) + len(new_patient_ids)
            desired = _target_counts(total_after)
            # New patients are ordered by a seeded hash, giving deterministic
            # splits without relying on filesystem or manifest ordering.
            ordered = sorted(
                new_patient_ids,
                key=lambda patient_id: _stable_order(self.seed, label, patient_id),
            )
            for patient_id in ordered:
                split = _choose_split(current_counts, desired)
                current_counts[split] += 1
                result[patient_id] = split
                self.assignments[patient_id] = {
                    "split": split,
                    "class_name": label,
                    "assigned_at": datetime.now(timezone.utc).isoformat(),
                }

        return result

    def save(self) -> None:
        """Persist split assignments atomically.

        The registry is written to a temporary file first and then replaced so
        interrupted writes do not leave a partially written JSON file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "seed": self.seed,
            "assignments": dict(sorted(self.assignments.items())),
        }
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self.path)


def _target_counts(total: int) -> dict[str, int]:
    """Return desired split counts for a label at a given total.

    Args:
        total: Number of patients for one class label after new assignments.

    Returns:
        Desired train, validation, and test patient counts.
    """
    train = int(total * TARGET_RATIOS["train"])
    validation = int(total * TARGET_RATIOS["validation"])
    return {
        "train": train,
        "validation": validation,
        "test": total - train - validation,
    }


def _choose_split(current: Counter, desired: dict[str, int]) -> str:
    """Choose the split with the largest current deficit.

    Args:
        current: Current split counts for one class label.
        desired: Desired split counts for that label.

    Returns:
        Split name that best restores the target ratio.
    """
    deficits = {split: desired[split] - current[split] for split in SPLITS}
    positive = [split for split in SPLITS if deficits[split] > 0]
    if positive:
        return max(positive, key=lambda split: (deficits[split], -SPLITS.index(split)))
    return min(
        SPLITS,
        key=lambda split: (
            current[split] / max(desired[split], 1),
            SPLITS.index(split),
        ),
    )


def _stable_order(seed: str, label: str, patient_id: str) -> str:
    """Return a deterministic ordering key for a patient and class label.

    Args:
        seed: Project split seed.
        label: Patient class label.
        patient_id: Stable patient identifier.

    Returns:
        SHA-256 hex digest used as a deterministic sort key.
    """
    value = f"{seed}|{label}|{patient_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()
