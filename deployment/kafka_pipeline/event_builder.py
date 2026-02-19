from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_prediction_event(
    *,
    request_id: str,
    input_type: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_type": "prediction.completed",
        "event_version": "1.0",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "input_type": input_type,
        "payload": response,
    }
