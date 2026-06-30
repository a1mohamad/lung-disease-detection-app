"""Kafka consumer that builds a doctor-review image queue."""

from __future__ import annotations

from pathlib import Path

from app.configs.config import AppConfig
from kafka_pipeline.consumers.common import append_jsonl, build_consumer, poll_event

OUT = Path("runtime/doctor_queue.jsonl")


def main() -> None:
    """Consume prediction events and append review-ready artifact links.

    The review queue focuses on image artifacts and labels, not model internals.
    That makes it a clean handoff for a future clinician-facing workflow.
    """
    if not AppConfig.KAFKA_ENABLED:
        print("KAFKA_ENABLED=false, doctor-images consumer stopped.")
        return

    consumer = build_consumer(AppConfig.KAFKA_GROUP_DOCTOR)
    print("Doctor images consumer started.")
    try:
        while True:
            event = poll_event(consumer)
            if not event:
                continue

            payload = event["payload"]
            # Store both URLs and storage paths: URLs help human review, while
            # paths remain stable after signed URLs expire.
            out = {
                "request_id": event.get("request_id"),
                "occurred_at": event.get("occurred_at"),
                "final_label_name": payload.get("final_label_name"),
                "source_url": payload.get("source_url"),
                "mask_url": payload.get("mask_url"),
                "roi_url": payload.get("roi_url"),
                "overlay_url": payload.get("overlay_url"),
                "source_path": payload.get("source_path"),
                "mask_path": payload.get("mask_path"),
                "roi_path": payload.get("roi_path"),
                "overlay_path": payload.get("overlay_path"),
            }
            append_jsonl(OUT, out)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
