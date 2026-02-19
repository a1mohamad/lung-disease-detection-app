from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.configs.config import AppConfig
from kafka_pipeline.consumers.common import append_jsonl, build_consumer, poll_event

OUT = Path("runtime/monitoring_metrics.jsonl")


def main() -> None:
    if not AppConfig.KAFKA_ENABLED:
        print("KAFKA_ENABLED=false, monitoring consumer stopped.")
        return

    consumer = build_consumer(AppConfig.KAFKA_GROUP_MONITORING)
    window: deque[dict] = deque()
    print("Monitoring consumer started.")
    try:
        while True:
            event = poll_event(consumer)
            if not event:
                continue

            now = datetime.now(timezone.utc)
            payload = event["payload"]
            window.append({"at": now, "final_label": payload.get("final_label"), "final_prob": payload.get("final_prob", 0.0)})

            cutoff = now - timedelta(minutes=5)
            while window and window[0]["at"] < cutoff:
                window.popleft()

            total = len(window)
            unhealthy = sum(1 for x in window if x["final_label"] == 1)
            avg_prob = (sum(float(x["final_prob"]) for x in window) / total) if total else 0.0
            metrics = {
                "at": now.isoformat(),
                "window_minutes": 5,
                "requests_count": total,
                "unhealthy_rate": (unhealthy / total) if total else 0.0,
                "avg_final_prob": avg_prob,
            }
            append_jsonl(OUT, metrics)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
