from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mlops.core.ingestion.manifest import ReviewedRecord, load_reviewed_batch
from mlops.core.ingestion.splits import SplitRegistry


class ReviewedManifestTests(unittest.TestCase):
    def test_valid_manifest_loads(self):
        payload = {
            "schema_version": "1.0",
            "batch_id": "2026-06",
            "period_start": "2026-06-01T00:00:00Z",
            "period_end": "2026-07-01T00:00:00Z",
            "records": [
                {
                    "sample_id": "sample-1",
                    "patient_id": "patient-1",
                    "image_key": "batches/2026/06/images/sample-1.png",
                    "mask_key": "batches/2026/06/masks/sample-1.png",
                    "class_name": "COVID",
                    "reviewed_at": "2026-06-15T00:00:00Z",
                }
            ],
        }
        batch = load_reviewed_batch(
            json.dumps(payload).encode("utf-8"),
            source_id="manifest.json",
        )
        self.assertEqual(batch.batch_id, "2026-06")
        self.assertEqual(len(batch.records), 1)

    def test_review_timestamp_outside_batch_is_rejected(self):
        payload = {
            "schema_version": "1.0",
            "batch_id": "2026-06",
            "period_start": "2026-06-01T00:00:00Z",
            "period_end": "2026-07-01T00:00:00Z",
            "records": [
                {
                    "sample_id": "sample-1",
                    "patient_id": "patient-1",
                    "image_key": "image.png",
                    "mask_key": "mask.png",
                    "class_name": "Normal",
                    "reviewed_at": "2026-07-02T00:00:00Z",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "within the batch period"):
            load_reviewed_batch(
                json.dumps(payload).encode("utf-8"),
                source_id="manifest.json",
            )


class StableSplitTests(unittest.TestCase):
    def test_balanced_assignment_and_permanence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split-registry.json"
            registry = SplitRegistry.load(path, "test-seed")
            first_records = _records("COVID", 100)
            first_assignments = registry.assign(first_records)
            registry.save()

            self.assertEqual(
                Counter(first_assignments.values()),
                Counter({"train": 80, "validation": 10, "test": 10}),
            )

            original = dict(first_assignments)
            registry = SplitRegistry.load(path, "test-seed")
            combined = first_records + _records("COVID", 20, offset=100)
            second_assignments = registry.assign(combined)

            for patient_id, split in original.items():
                self.assertEqual(second_assignments[patient_id], split)
            self.assertEqual(
                Counter(second_assignments.values()),
                Counter({"train": 96, "validation": 12, "test": 12}),
            )

    def test_all_samples_for_patient_use_one_split(self):
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        records = [
            ReviewedRecord(
                sample_id=f"sample-{index}",
                patient_id="same-patient",
                image_key=f"images/{index}.png",
                mask_key=f"masks/{index}.png",
                class_name="Normal",
                reviewed_at=now,
            )
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = SplitRegistry.load(
                Path(temp_dir) / "split-registry.json",
                "test-seed",
            )
            assignments = registry.assign(records)
        self.assertEqual(set(assignments), {"same-patient"})


def _records(class_name: str, count: int, offset: int = 0) -> list[ReviewedRecord]:
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    return [
        ReviewedRecord(
            sample_id=f"sample-{index}",
            patient_id=f"patient-{index}",
            image_key=f"images/{index}.png",
            mask_key=f"masks/{index}.png",
            class_name=class_name,
            reviewed_at=now,
        )
        for index in range(offset, offset + count)
    ]


if __name__ == "__main__":
    unittest.main()
