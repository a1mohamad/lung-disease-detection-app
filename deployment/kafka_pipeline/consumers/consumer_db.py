from __future__ import annotations
import warnings
from sqlalchemy.exc import SAWarning

import os
from pathlib import Path

from dotenv import dotenv_values

warnings.filterwarnings(
    "ignore",
    message=".*Unrecognized server version info.*",
    category=SAWarning,
)


# Load project env before importing DB modules (engine config is read at import time).
_env_path = Path(__file__).resolve().parents[2] / ".env"
for _k, _v in dotenv_values(_env_path).items():
    if _v is not None:
        os.environ[_k] = _v

from app.configs.config import AppConfig
from app.db.crud import log_prediction
from app.db.session import SessionLocal
from kafka_pipeline.consumers.common import build_consumer, poll_event


def main() -> None:
    if not AppConfig.KAFKA_ENABLED:
        print("KAFKA_ENABLED=false, db consumer stopped.")
        return

    consumer = build_consumer(AppConfig.KAFKA_GROUP_DB)
    print("DB consumer started.")
    try:
        while True:
            event = poll_event(consumer)
            if not event:
                continue

            db = SessionLocal()
            try:
                log_prediction(
                    db=db,
                    request_id=event["request_id"],
                    input_type=event["input_type"],
                    response=event["payload"],
                )
            finally:
                db.close()
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
