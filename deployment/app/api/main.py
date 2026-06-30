"""FastAPI application factory and runtime lifecycle hooks."""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.api.startup import check_paths_and_metadata, create_detector, init_database, warmup
from app.configs.config import AppConfig
from app.db.session import engine
from fastapi.staticfiles import StaticFiles
from kafka_pipeline.producer import close_kafka_producer, init_kafka_producer


# TensorFlow is imported indirectly by the predictor stack, so logging is
# quieted before the application imports model-loading modules.
class SPAStaticFiles(StaticFiles):
    """Serve a static single-page application alongside the FastAPI backend.

    Browser refreshes on client-side routes such as ``/ui/history`` should load
    the frontend entry point instead of returning a backend 404. Asset requests
    still use the normal ``StaticFiles`` behavior.
    """

    async def get_response(self, path: str, scope):
        """Return a static asset or fall back to ``index.html`` for SPA routes.

        Args:
            path: Requested static path.
            scope: ASGI request scope supplied by Starlette.

        Returns:
            Static file response for assets, or the SPA entry point for
            client-side routes.
        """
        try:
            return await super().get_response(path, scope)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return await super().get_response("index.html", scope)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up process-wide API resources.

    Startup validates artifacts, initializes optional persistence/eventing
    integrations, creates the model pipeline, and performs a warmup prediction
    before serving traffic. Shutdown flushes Kafka and disposes database engine
    resources when they exist.
    """
    check_paths_and_metadata()
    # Database and Kafka are optional integrations. Startup should still serve
    # local inference when those services are disabled or unavailable.
    try:
        init_database()
    except Exception as exc:
        logging.getLogger(__name__).warning("Database init skipped: %s", exc)
    try:
        init_kafka_producer()
    except Exception as exc:
        logging.getLogger(__name__).warning("Kafka init skipped: %s", exc)
    app.state.detector = create_detector()
    # Warmup catches artifact/runtime problems before the first user request
    # and removes the cold-start penalty from the first real prediction.
    warmup(app.state.detector)
    yield
    # Flush external resources explicitly so container shutdowns do not drop
    # pending Kafka events or leave SQLAlchemy pools open.
    close_kafka_producer()
    if engine is not None:
        engine.dispose()

app = FastAPI(
    title="Lung Disease Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)
if AppConfig.CORS_ALLOW_ORIGINS:
    # CORS is environment-driven to keep local UI development flexible without
    # widening browser access in locked-down deployments.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=AppConfig.CORS_ALLOW_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
app.include_router(router)
# Generated masks, ROI crops, overlays, and source previews are served under a
# stable static prefix so API responses can link directly to review artifacts.
app.mount("/static", StaticFiles(directory=AppConfig.ASSETS_DIR), name="static")
if AppConfig.FRONTEND_DIR.exists():
    # The bundled frontend is optional; API-only deployments can omit it without
    # changing the backend image or health checks.
    app.mount("/ui", SPAStaticFiles(directory=AppConfig.FRONTEND_DIR, html=True), name="ui")
