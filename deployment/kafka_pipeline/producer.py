"""Kafka producer lifecycle and prediction event publishing."""

from __future__ import annotations

import json
from typing import Any

from app.configs.config import AppConfig

_producer: Any | None = None


def _delivery_report(err, msg) -> None:
    """Report asynchronous Kafka delivery failures.

    Args:
        err: Kafka delivery error, if any.
        msg: Kafka message metadata supplied by the producer callback.
    """
    if err is not None:
        print(f"[kafka] delivery failed: {err}")


def _producer_config() -> dict[str, str]:
    """Build producer configuration from application settings.

    Returns:
        Confluent Kafka producer configuration dictionary.
    """
    security_protocol = AppConfig.KAFKA_SECURITY_PROTOCOL
    conf = {
        "bootstrap.servers": AppConfig.KAFKA_BOOTSTRAP_SERVERS,
        "client.id": AppConfig.KAFKA_CLIENT_ID,
        "security.protocol": security_protocol,
    }

    # Local Compose uses PLAINTEXT, while cloud Kafka providers often require
    # SASL. Only attach credentials when the selected protocol needs them.
    if "SASL" not in security_protocol.upper():
        return conf

    if AppConfig.KAFKA_SASL_MECHANISM:
        conf["sasl.mechanism"] = AppConfig.KAFKA_SASL_MECHANISM
    if AppConfig.KAFKA_SASL_USERNAME:
        conf["sasl.username"] = AppConfig.KAFKA_SASL_USERNAME
    if AppConfig.KAFKA_SASL_PASSWORD:
        conf["sasl.password"] = AppConfig.KAFKA_SASL_PASSWORD
    return conf


def init_kafka_producer() -> None:
    """Initialize the process-wide Kafka producer when enabled.

    Notes:
        The producer is created lazily so tests and non-Kafka deployments do not
        import or connect to Kafka unnecessarily.
    """
    global _producer
    if not AppConfig.KAFKA_ENABLED:
        return
    if _producer is None:
        from confluent_kafka import Producer

        _producer = Producer(_producer_config())


def publish_prediction_event(*, request_id: str, event: dict[str, Any]) -> None:
    """Publish one prediction event to the configured Kafka topic.

    Args:
        request_id: Stable event key used for partitioning and log correlation.
        event: JSON-serializable prediction payload built by ``event_builder``.

    Notes:
        The API route catches publish failures so inference can still return a
        response even when the event pipeline is temporarily unavailable.
    """
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
    """Flush and release the process-wide Kafka producer.

    Shutdown flushing gives queued delivery callbacks a chance to finish before
    the FastAPI process exits.
    """
    global _producer
    if _producer is not None:
        _producer.flush(10)
        _producer = None
