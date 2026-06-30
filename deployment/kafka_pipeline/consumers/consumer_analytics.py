"""Kafka consumer that writes compact analytics records."""

from __future__ import annotations

from pathlib import Path

from app.configs.config import AppConfig
from kafka_pipeline.consumers.common import append_jsonl, build_consumer, poll_event

OUT = Path("runtime/analytics_events.jsonl")


def _best_model(models: dict) -> tuple[str | None, float | None]:
    """Return the binary model with the highest unhealthy probability.

    Args:
        models: Per-model ensemble result dictionary from the API response.

    Returns:
        Tuple of ``(model_name, probability)`` for the strongest unhealthy
        signal, or ``(None, None)`` when no valid model entries exist.
    """
    best_name = None
    best_prob = None
    for name, result in models.items():
        if not isinstance(result, dict):
            continue
        prob = float(result.get("prob", 0.0))
        if best_prob is None or prob > best_prob:
            best_prob = prob
            best_name = name
    return best_name, best_prob


def main() -> None:
    """Consume prediction events and append analytics JSONL records.

    The analytics consumer writes compact derived records rather than the full
    prediction payload. This keeps downstream dashboards lightweight while the
    database consumer preserves the full normalized prediction log.
    """
    if not AppConfig.KAFKA_ENABLED:
        print("KAFKA_ENABLED=false, analytics consumer stopped.")
        return

    consumer = build_consumer(AppConfig.KAFKA_GROUP_ANALYTICS)
    print("Analytics consumer started.")
    try:
        while True:
            event = poll_event(consumer)
            if not event:
                continue

            payload = event["payload"]
            models = payload.get("models_results", {})
            # Store a simple "strongest model" signal so model disagreement can
            # be tracked later without reprocessing the full event payload.
            best_name, best_prob = _best_model(models if isinstance(models, dict) else {})
            out = {
                "occurred_at": event.get("occurred_at"),
                "request_id": event.get("request_id"),
                "final_label": payload.get("final_label"),
                "final_label_name": payload.get("final_label_name"),
                "final_prob": payload.get("final_prob"),
                "best_model_name": best_name,
                "best_model_prob": best_prob,
                "models_count": len(models) if isinstance(models, dict) else 0,
                "disease_label_name": (payload.get("disease") or {}).get("label_name"),
            }
            append_jsonl(OUT, out)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
