# Kafka Pipeline

[![Kafka](https://img.shields.io/badge/Kafka-Prediction%20Events-231F20?logo=apachekafka&logoColor=white)](https://kafka.apache.org/)
[![JSON](https://img.shields.io/badge/Event%20Format-JSON-lightgrey)](https://www.json.org/json-en.html)
[![Consumers](https://img.shields.io/badge/Consumers-5-blue)](#consumers)
[![Mode](https://img.shields.io/badge/Runtime-Optional-success)](#purpose)

**Cloud Integration Targets**

[![Render PostgreSQL](https://img.shields.io/badge/Render-DB%20Consumer%20Target-46E3B7?logo=render&logoColor=black)](https://render.com/docs/databases)
[![Supabase Storage](https://img.shields.io/badge/Supabase-Artifact%20Links-3FCF8E?logo=supabase&logoColor=white)](https://supabase.com/storage)
[![Research Lab](https://img.shields.io/badge/Research%20Lab-Project%20Page-222222?logo=githubpages&logoColor=white)](https://a1mohamad.github.io/research/lung-disease-detection/index.html)

Optional event-driven layer for prediction persistence, analytics, monitoring, doctor-review queues, and notifications.

---

## Table of Contents

- [Purpose](#purpose)
- [Event Flow](#event-flow)
- [Event Contract](#event-contract)
- [Producer](#producer)
- [Consumers](#consumers)
- [Configuration](#configuration)
- [Running Consumers](#running-consumers)
- [Design Notes](#design-notes)

---

## Purpose

The API can persist predictions directly to the database, but Kafka makes the system more flexible:

- inference remains focused on prediction
- database writes can be handled asynchronously
- analytics can evolve independently
- monitoring can run as a separate consumer
- doctor-review and notification workflows do not block user requests

When `KAFKA_ENABLED=true`, the API publishes one event per completed prediction. When `KAFKA_ENABLED=false`, the API can log directly to the database if `DB_LOGGING_ENABLED=true`.

The current public runtime keeps Kafka optional and uses direct persistence to managed PostgreSQL. The Kafka package remains useful for local/full-stack demonstrations where prediction logging, analytics, monitoring, review queues, and notifications are split into independent consumers.

The related research context is available in the [lung disease detection research lab](https://a1mohamad.github.io/research/lung-disease-detection/index.html).

---

## Event Flow

```text
FastAPI prediction route
    |
    v
build_prediction_event()
    |
    v
publish_prediction_event()
    |
    v
Kafka topic: lung.predictions
    |
    |-- DB consumer
    |-- analytics consumer
    |-- monitoring consumer
    |-- doctor image queue consumer
    +-- notification consumer
```

---

## Event Contract

```json
{
  "event_type": "prediction.completed",
  "event_version": "1.0",
  "occurred_at": "2026-06-29T00:00:00+00:00",
  "request_id": "0fb6f7a6-9e4e-4d9b-8bb2-9a1d8d65efde",
  "input_type": "upload",
  "payload": {
    "final_prob": 0.84,
    "final_probs_by_label": {
      "healthy": 0.16,
      "unhealthy": 0.84
    },
    "final_label": 1,
    "final_label_name": "Unhealthy",
    "source_url": "/static/predictions/2026-06-29/id/source.png",
    "mask_url": "/static/predictions/2026-06-29/id/mask.png",
    "roi_url": "/static/predictions/2026-06-29/id/roi.png",
    "overlay_url": "/static/predictions/2026-06-29/id/overlay.png"
  }
}
```

The `payload` is the same dictionary returned by the prediction API.

---

## Producer

`producer.py` owns:

- process-wide producer initialization
- SASL-aware Kafka configuration
- event JSON serialization
- `request_id` message keys
- shutdown flushing

The API initializes the producer during startup and flushes it during shutdown.

---

## Consumers

| Consumer | Output | Use Case |
|---|---|---|
| `consumer_db.py` | SQLAlchemy database rows | durable prediction history |
| `consumer_analytics.py` | `runtime/analytics_events.jsonl` | aggregate reporting and dashboards |
| `consumer_monitoring.py` | `runtime/monitoring_metrics.jsonl` | rolling operational metrics |
| `consumer_doctor_images.py` | `runtime/doctor_queue.jsonl` | review queue with generated artifact links |
| `consumer_notifications.py` | `runtime/notifications_outbox.jsonl` | user-facing message queue |

For a cloud deployment that enables Kafka, `consumer_db.py` can target the same managed Postgres database used by the direct logging path, while artifact URLs can point to Supabase-signed prediction images.

---

## Configuration

| Variable | Default | Description |
|---|---:|---|
| `KAFKA_ENABLED` | `true` | Enables producer and consumers |
| `KAFKA_BOOTSTRAP_SERVERS` | `127.0.0.1:9092` | Broker address list |
| `KAFKA_TOPIC_PREDICTIONS` | `lung.predictions` | Prediction event topic |
| `KAFKA_CLIENT_ID` | `lung-api-producer` | Producer client id |
| `KAFKA_SECURITY_PROTOCOL` | `PLAINTEXT` | Kafka security protocol |
| `KAFKA_SASL_MECHANISM` | empty | SASL mechanism when needed |
| `KAFKA_SASL_USERNAME` | empty | SASL username |
| `KAFKA_SASL_PASSWORD` | empty | SASL password |

Consumer groups:

| Variable | Default |
|---|---|
| `KAFKA_GROUP_DB` | `lung-consumer-db` |
| `KAFKA_GROUP_MONITORING` | `lung-consumer-monitoring` |
| `KAFKA_GROUP_ANALYTICS` | `lung-consumer-analytics` |
| `KAFKA_GROUP_DOCTOR` | `lung-consumer-doctor` |
| `KAFKA_GROUP_NOTIFICATIONS` | `lung-consumer-notifications` |

---

## Running Consumers

From `deployment`:

```powershell
$env:PYTHONPATH="."
python -m kafka_pipeline.consumers.consumer_db
python -m kafka_pipeline.consumers.consumer_analytics
python -m kafka_pipeline.consumers.consumer_monitoring
python -m kafka_pipeline.consumers.consumer_doctor_images
python -m kafka_pipeline.consumers.consumer_notifications
```

---

## Design Notes

- The DB consumer owns persistence when Kafka is enabled to avoid duplicate writes.
- Consumer startup tolerates temporary broker/topic warmup failures.
- JSONL consumers are simple integration points that can later be replaced by dashboards, alerting systems, or notification services.
- The event schema includes `event_version` so future payload changes can be introduced safely.
