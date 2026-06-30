"""Shared Kafka consumer utilities for prediction event consumers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

# Load deployment/.env only for local direct runs; keep container-provided env values.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def consumer_config(group_id: str) -> dict[str, str]:
    """Build a Kafka consumer configuration for a consumer group.

    Args:
        group_id: Kafka consumer group id. Each consumer type uses a different
            group so all workflows receive every prediction event.

    Returns:
        Confluent Kafka consumer configuration dictionary.
    """
    security_protocol = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092"),
        "group.id": group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": "true",
        "security.protocol": security_protocol,
    }
    if "SASL" not in security_protocol.upper():
        return conf

    if os.getenv("KAFKA_SASL_MECHANISM"):
        conf["sasl.mechanism"] = os.getenv("KAFKA_SASL_MECHANISM", "")
    if os.getenv("KAFKA_SASL_USERNAME"):
        conf["sasl.username"] = os.getenv("KAFKA_SASL_USERNAME", "")
    if os.getenv("KAFKA_SASL_PASSWORD"):
        conf["sasl.password"] = os.getenv("KAFKA_SASL_PASSWORD", "")
    return conf


def build_consumer(group_id: str) -> Consumer:
    """Create and subscribe a Kafka consumer for prediction events.

    Args:
        group_id: Consumer group id configured for this workflow.

    Returns:
        Subscribed Kafka consumer instance.
    """
    c = Consumer(consumer_config(group_id))
    c.subscribe([os.getenv("KAFKA_TOPIC_PREDICTIONS", "lung.predictions")])
    return c


def poll_event(consumer: Consumer) -> dict[str, Any] | None:
    """Poll one prediction event and normalize transient broker errors.

    Returns:
        Decoded event dictionary, or ``None`` when there is no message or the
        broker is in a recoverable startup state.

    Raises:
        RuntimeError: If Kafka reports a non-transient consumer error.
    """
    msg = consumer.poll(1.0)
    if msg is None:
        return None
    if msg.error():
        code = msg.error().code()
        # During startup the topic may not exist yet or brokers may still be warming up.
        # Treat these as transient so long-running consumers don't crash.
        if code in {
            KafkaError._PARTITION_EOF,
            KafkaError.UNKNOWN_TOPIC_OR_PART,
            KafkaError._TRANSPORT,
            KafkaError._ALL_BROKERS_DOWN,
            KafkaError.REQUEST_TIMED_OUT,
        }:
            return None
        raise RuntimeError(f"Kafka consumer error: {msg.error()}")
    return json.loads(msg.value().decode("utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a JSON payload to a local JSON Lines file.

    Args:
        path: Output JSONL path.
        payload: JSON-serializable event-derived record.

    Notes:
        JSONL keeps local demo consumers inspectable without introducing another
        database for analytics, monitoring, or notification outboxes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
