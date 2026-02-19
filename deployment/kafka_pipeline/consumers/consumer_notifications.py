from __future__ import annotations

from pathlib import Path

from app.configs.config import AppConfig
from kafka_pipeline.consumers.common import append_jsonl, build_consumer, poll_event

OUT = Path("runtime/notifications_outbox.jsonl")


def _final_selected_prob(payload: dict) -> float | None:
    probs = payload.get("final_probs_by_label")
    if not isinstance(probs, dict):
        return None
    label_name = payload.get("final_label_name")
    if isinstance(label_name, str) and label_name in probs:
        return float(probs[label_name])
    return None


def main() -> None:
    if not AppConfig.KAFKA_ENABLED:
        print("KAFKA_ENABLED=false, notifications consumer stopped.")
        return

    consumer = build_consumer(AppConfig.KAFKA_GROUP_NOTIFICATIONS)
    print("Notifications consumer started.")
    try:
        while True:
            event = poll_event(consumer)
            if not event:
                continue

            payload = event["payload"]
            final_label_name = payload.get("final_label_name")
            selected_prob = _final_selected_prob(payload)
            disease = payload.get("disease") if isinstance(payload.get("disease"), dict) else None
            disease_txt = ""
            if disease and disease.get("label_name"):
                disease_txt = f" Disease likely: {disease.get('label_name')}."

            msg = (
                f"Prediction result: {final_label_name}. "
                f"Confidence: {selected_prob:.2%}." if selected_prob is not None else f"Prediction result: {final_label_name}."
            )
            msg += disease_txt
            out = {
                "request_id": event.get("request_id"),
                "occurred_at": event.get("occurred_at"),
                "message": msg,
            }
            append_jsonl(OUT, out)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
