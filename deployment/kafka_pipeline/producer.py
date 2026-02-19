from __future__ import annotations

import json
from typing import Any

from app.configs.config import AppConfig

_producer: Any | None = None


def _delivery_report(err, msg) -> None:
    if err is not None:
        print(f"[kafka] delivery failed: {err}")


def _producer_config() -> dict[str, str]:
    conf = {
        "bootstrap.servers": AppConfig.KAFKA_BOOTSTRAP_SERVERS,
        "client.id": AppConfig.KAFKA_CLIENT_ID,
        "security.protocol": AppConfig.KAFKA_SECURITY_PROTOCOL,
    }

    if AppConfig.KAFKA_SASL_MECHANISM:
        conf["sasl.mechanism"] = AppConfig.KAFKA_SASL_MECHANISM
    if AppConfig.KAFKA_SASL_USERNAME:
        conf["sasl.username"] = AppConfig.KAFKA_SASL_USERNAME
    if AppConfig.KAFKA_SASL_PASSWORD:
        conf["sasl.password"] = AppConfig.KAFKA_SASL_PASSWORD
    return conf


def init_kafka_producer() -> None:
    global _producer
    if not AppConfig.KAFKA_ENABLED:
        return
    if _producer is None:
        from confluent_kafka import Producer

        _producer = Producer(_producer_config())


def publish_prediction_event(*, request_id: str, event: dict[str, Any]) -> None:
    if not AppConfig.KAFKA_ENABLED:
        return
    init_kafka_producer()
    assert _producer is not None
    _producer.produce(
        topic=AppConfig.KAFKA_TOPIC_PREDICTIONS,
        key=request_id,
        value=json.dumps(event, ensure_ascii=True).encode("utf-8"),
        callback=_delivery_report,
    )
    _producer.poll(0)


def close_kafka_producer() -> None:
    global _producer
    if _producer is not None:
        _producer.flush(10)
        _producer = None
