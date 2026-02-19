import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.api.startup import check_paths_and_metadata, create_detector, init_database, warmup
from app.configs.config import AppConfig
from app.db.session import engine
from fastapi.staticfiles import StaticFiles
from kafka_pipeline.producer import close_kafka_producer, init_kafka_producer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: place any init logic here
    check_paths_and_metadata()
    try:
        init_database()
    except Exception as exc:
        logging.getLogger(__name__).warning("Database init skipped: %s", exc)
    try:
        init_kafka_producer()
    except Exception as exc:
        logging.getLogger(__name__).warning("Kafka init skipped: %s", exc)
    app.state.detector = create_detector()
    warmup(app.state.detector)
    yield
    # shutdown: cleanup if needed
    close_kafka_producer()
    engine.dispose()

app = FastAPI(
    title="Lung Disease Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
app.include_router(router)
app.mount("/static", StaticFiles(directory=AppConfig.ASSETS_DIR), name="static")
