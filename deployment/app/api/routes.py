"""HTTP route handlers for health checks, prediction, and prediction logs."""

import logging
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile
from sqlalchemy.orm import Session

from app.configs.config import AppConfig
from app.db.crud import get_prediction_logs, log_prediction
from app.db.session import SessionLocal, get_db
from app.schemas.health import HealthResponse
from app.schemas.logs import PredictionLogsResponse
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.inference import run_inference
from app.utils.errors import AuthError, InputError, ServiceError
from kafka_pipeline.event_builder import build_prediction_event
from kafka_pipeline.producer import publish_prediction_event

router = APIRouter(tags=["predictions"])
logger = logging.getLogger(__name__)


def _detect_input_type(req: PredictRequest) -> str:
    """Return the request input source name used by logs and Kafka events.

    Args:
        req: Validated JSON prediction request.

    Returns:
        Source label such as ``path``, ``url``, ``base64``, or ``unknown``.
    """
    if req.image_path:
        return "path"
    if req.image_url:
        return "url"
    if req.image_base64:
        return "base64"
    return "unknown"


def _persist_prediction_log(*, request_id: str, input_type: str, response: dict) -> None:
    """Persist a prediction directly when the Kafka pipeline is disabled.

    With Kafka enabled the DB consumer owns this write, so callers must gate on
    KAFKA_ENABLED to avoid double-logging the same prediction.

    Args:
        request_id: UUID generated for the API request.
        input_type: Source category used for analytics and auditing.
        response: API prediction response to normalize into relational tables.

    Notes:
        Persistence failures are logged and swallowed because prediction logging
        is an operational side effect, not the primary inference result.
    """
    if SessionLocal is None:
        return
    db = SessionLocal()
    try:
        log_prediction(
            db=db,
            request_id=request_id,
            input_type=input_type,
            response=response,
        )
    except Exception as exc:
        logger.exception("Direct DB log failed: %s", exc)
    finally:
        db.close()


def _authorize_logs(x_api_key: str | None) -> None:
    """Validate access to the protected prediction-log endpoint.

    Args:
        x_api_key: Value of the ``X-API-Key`` request header.

    Raises:
        ServiceError: If log access is disabled by configuration.
        AuthError: If the provided key does not match the configured key.
    """
    if not AppConfig.LOGS_API_KEY:
        raise ServiceError("LOGS_DISABLED", "Logs endpoint is disabled.")
    if x_api_key != AppConfig.LOGS_API_KEY:
        raise AuthError("INVALID_API_KEY", "Invalid API key.")


@router.get("/health", response_model=HealthResponse)
def health():
    """Return a lightweight liveness response for load balancers and CI checks.

    Returns:
        Service status and API version.
    """
    return {"status": "ok", "version": "1.0.0"}


@router.get("/", response_model=HealthResponse)
def root():
    """Return API identity information and basic health status.

    Returns:
        API name, status, and version.
    """
    return {"name": "Lung Disease Detection API", "status": "ok", "version": "1.0.0"}


@router.post("/predict", response_model=PredictResponse, status_code=200)
def predict_json(
    request: Request, 
    req: PredictRequest, 
    return_all: bool = True
    ):
    """Run inference from a JSON image source.

    The request body must contain exactly one supported source: local path,
    remote URL, or base64 payload. The returned prediction is also emitted to
    Kafka when enabled, otherwise it is written directly to the database when
    database logging is enabled.

    Args:
        request: FastAPI request carrying the process-wide detector.
        req: Validated prediction request payload.
        return_all: Include individual ensemble-member results when true.

    Returns:
        Public prediction response with probabilities, labels, optional disease
        subtype, and generated artifact links.
    """
    request_id = str(uuid4())
    response = run_inference(
        detector=request.app.state.detector,
        image_path=req.image_path,
        image_base64=req.image_base64,
        image_url=req.image_url,
        return_all=return_all,
    )

    if AppConfig.KAFKA_ENABLED:
        try:
            event = build_prediction_event(
                request_id=request_id,
                input_type=_detect_input_type(req),
                response=response,
            )
            publish_prediction_event(request_id=request_id, event=event)
        except Exception as exc:
            # Kafka is an async integration boundary; inference responses should
            # still reach the client if event publication is temporarily down.
            logger.exception("Kafka publish failed: %s", exc)
    elif AppConfig.DB_LOGGING_ENABLED:
        _persist_prediction_log(
            request_id=request_id,
            input_type=_detect_input_type(req),
            response=response,
        )

    return response


@router.post("/predict/upload", response_model=PredictResponse, status_code=200)
async def predict_upload(
    request: Request, 
    file: UploadFile = File(...), 
    return_all: bool = True
    ):
    """Run inference from a multipart image upload.

    This endpoint mirrors ``/predict`` but accepts file streams instead of JSON
    sources. It uses the same inference service and logging/eventing behavior so
    both API styles produce the same response contract.

    Args:
        request: FastAPI request carrying the process-wide detector.
        file: Uploaded image file stream.
        return_all: Include individual ensemble-member results when true.

    Returns:
        Public prediction response with probabilities, labels, optional disease
        subtype, and generated artifact links.
    """
    request_id = str(uuid4())
    response = run_inference(
        detector=request.app.state.detector,
        upload_file=file.file,
        return_all=return_all,
    )

    if AppConfig.KAFKA_ENABLED:
        try:
            event = build_prediction_event(
                request_id=request_id,
                input_type="upload",
                response=response,
            )
            publish_prediction_event(request_id=request_id, event=event)
        except Exception as exc:
            # Keep upload inference available even when Kafka is degraded.
            logger.exception("Kafka publish failed: %s", exc)
    elif AppConfig.DB_LOGGING_ENABLED:
        _persist_prediction_log(
            request_id=request_id,
            input_type="upload",
            response=response,
        )

    return response


@router.get("/logs", response_model=PredictionLogsResponse)
def get_logs(
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    x_api_key: str | None = Header(default=None),
):
    """Return paginated prediction logs for authorized operational review.

    Args:
        db: SQLAlchemy session provided by FastAPI dependency injection.
        limit: Maximum number of records to return, constrained to 1..200.
        offset: Number of records to skip.
        x_api_key: API key header required for access.

    Returns:
        Normalized prediction log response including child model, disease, and
        image-link records.

    Raises:
        AuthError: If the API key is invalid.
        InputError: If pagination parameters are outside supported bounds.
    """
    _authorize_logs(x_api_key)

    if limit < 1 or limit > 200:
        raise InputError("INVALID_LIMIT", "limit must be between 1 and 200")
    if offset < 0:
        raise InputError("INVALID_OFFSET", "offset must be >= 0")

    rows = get_prediction_logs(db=db, limit=limit, offset=offset)
    items = [
        {
            "id": r.id,
            "request_id": r.request_id,
            "input_type": r.input_type,
            "final_label": r.final_label,
            "final_label_name": r.final_label_name,
            "final_prob": r.final_prob,
            "final_probs_by_label": json.loads(r.final_probs_json) if r.final_probs_json else None,
            "binary_model_results": [
                {
                    "model_name": m.model_name,
                    "label": m.label,
                    "label_name": m.label_name,
                    "prob": m.prob,
                    "probs_by_label": json.loads(m.probs_json) if m.probs_json else None,
                }
                for m in r.binary_model_results
            ],
            "disease_result": (
                {
                    "label": r.disease_result.label,
                    "label_name": r.disease_result.label_name,
                    "probs_by_label": (
                        json.loads(r.disease_result.probs_json)
                        if r.disease_result and r.disease_result.probs_json
                        else None
                    ),
                }
                if r.disease_result
                else None
            ),
            "error_code": r.error_code,
            "error_message": r.error_message,
            "image_links": {
                "source_url": r.image_links.source_url if r.image_links else None,
                "mask_url": r.image_links.mask_url if r.image_links else None,
                "roi_url": r.image_links.roi_url if r.image_links else None,
                "overlay_url": r.image_links.overlay_url if r.image_links else None,
                "source_path": r.image_links.source_path if r.image_links else None,
                "mask_path": r.image_links.mask_path if r.image_links else None,
                "roi_path": r.image_links.roi_path if r.image_links else None,
                "overlay_path": r.image_links.overlay_path if r.image_links else None,
            },
            "created_at": r.created_at,
        }
        for r in rows
    ]
    return {"items": items, "limit": limit, "offset": offset}
