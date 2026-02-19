from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

# Ensure consumer processes load deployment/.env when run directly.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


def consumer_config(group_id: str) -> dict[str, str]:
    conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092"),
        "group.id": group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": "true",
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    }
    if os.getenv("KAFKA_SASL_MECHANISM"):
        conf["sasl.mechanism"] = os.getenv("KAFKA_SASL_MECHANISM", "")
    if os.getenv("KAFKA_SASL_USERNAME"):
        conf["sasl.username"] = os.getenv("KAFKA_SASL_USERNAME", "")
    if os.getenv("KAFKA_SASL_PASSWORD"):
        conf["sasl.password"] = os.getenv("KAFKA_SASL_PASSWORD", "")
    return conf


def build_consumer(group_id: str) -> Consumer:
    c = Consumer(consumer_config(group_id))
    c.subscribe([os.getenv("KAFKA_TOPIC_PREDICTIONS", "lung.predictions")])
    return c


def poll_event(consumer: Consumer) -> dict[str, Any] | None:
    msg = consumer.poll(1.0)
    if msg is None:
        return None
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            return None
        raise RuntimeError(f"Kafka consumer error: {msg.error()}")
    return json.loads(msg.value().decode("utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
