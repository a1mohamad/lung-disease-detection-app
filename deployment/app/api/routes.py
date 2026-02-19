import logging
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, Request, UploadFile
from sqlalchemy.orm import Session

from app.configs.config import AppConfig
from app.db.crud import get_prediction_logs
from app.db.session import get_db
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
    if req.image_path:
        return "path"
    if req.image_url:
        return "url"
    if req.image_base64:
        return "base64"
    return "unknown"


def _authorize_logs(x_api_key: str | None) -> None:
    if not AppConfig.LOGS_API_KEY:
        raise ServiceError("LOGS_DISABLED", "Logs endpoint is disabled.")
    if x_api_key != AppConfig.LOGS_API_KEY:
        raise AuthError("INVALID_API_KEY", "Invalid API key.")


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "version": "1.0.0"}


@router.get("/", response_model=HealthResponse)
def root():
    return {"name": "Lung Disease Detection API", "status": "ok", "version": "1.0.0"}


@router.post("/predict", response_model=PredictResponse, status_code=200)
def predict_json(
    request: Request, 
    req: PredictRequest, 
    return_all: bool = True
    ):
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
            logger.exception("Kafka publish failed: %s", exc)

    return response


@router.post("/predict/upload", response_model=PredictResponse, status_code=200)
async def predict_upload(
    request: Request, 
    file: UploadFile = File(...), 
    return_all: bool = True
    ):
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
            logger.exception("Kafka publish failed: %s", exc)

    return response


@router.get("/logs", response_model=PredictionLogsResponse)
def get_logs(
    db: Session = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    x_api_key: str | None = Header(default=None),
):
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
            },
            "created_at": r.created_at,
        }
        for r in rows
    ]
    return {"items": items, "limit": limit, "offset": offset}
