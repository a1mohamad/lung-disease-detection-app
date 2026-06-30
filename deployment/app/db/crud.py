"""Database persistence helpers for prediction logs."""

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    PredictionBinaryModelResult,
    PredictionDiseaseResult,
    PredictionImageLink,
    PredictionRequest,
)


def log_prediction(
    *,
    db: Session,
    request_id: str,
    input_type: str,
    response: dict[str, Any],
    error_code: str | None = None,
    error_message: str | None = None,
) -> PredictionRequest:
    """Persist a prediction response and all related child records.

    The prediction log is normalized across a parent request row plus optional
    image-link, per-model binary, and disease subtype rows. This keeps logs easy
    to query while preserving the full response contract returned by the API.

    Args:
        db: Active SQLAlchemy session.
        request_id: API request UUID or Kafka event key.
        input_type: Source category such as ``path``, ``url``, ``base64``, or
            ``upload``.
        response: Prediction response emitted by the inference service.
        error_code: Optional structured error code for failed events.
        error_message: Optional human-readable error message for failed events.

    Returns:
        Persisted parent ``PredictionRequest`` record.
    """
    disease = response.get("disease")
    final_probs = response.get("final_probs_by_label")
    record = PredictionRequest(
        request_id=request_id,
        input_type=input_type,
        final_label=response["final_label"],
        final_label_name=response.get("final_label_name"),
        final_prob=response["final_prob"],
        final_probs_json=json.dumps(final_probs, ensure_ascii=True) if isinstance(final_probs, dict) else None,
        error_code=error_code,
        error_message=error_message,
    )

    db.add(record)
    # Flush assigns the parent primary key before child rows are constructed,
    # while keeping the full normalized insert inside one transaction.
    db.flush()

    # Artifact links are separated from the request row because URLs and
    # storage paths evolve independently from model outputs.
    image_links = PredictionImageLink(
        prediction_request_id=record.id,
        source_url=response.get("source_url"),
        mask_url=response.get("mask_url"),
        roi_url=response.get("roi_url"),
        overlay_url=response.get("overlay_url"),
        source_path=response.get("source_path"),
        mask_path=response.get("mask_path"),
        roi_path=response.get("roi_path"),
        overlay_path=response.get("overlay_path"),
    )
    db.add(image_links)

    models_results = response.get("models_results")
    if isinstance(models_results, dict):
        # Each ensemble member is persisted as its own row so monitoring can
        # inspect disagreement and calibration trends per binary classifier.
        for model_name, model_data in models_results.items():
            if not isinstance(model_data, dict):
                continue
            probs = model_data.get("probs_by_label")
            db.add(
                PredictionBinaryModelResult(
                    prediction_request_id=record.id,
                    model_name=model_name,
                    label=model_data.get("label", -1),
                    label_name=model_data.get("label_name"),
                    prob=float(model_data.get("prob", 0.0)),
                    probs_json=json.dumps(probs, ensure_ascii=True) if isinstance(probs, dict) else None,
                )
            )

    if isinstance(disease, dict):
        # Disease output is optional and only exists for unhealthy predictions,
        # so it remains a nullable one-to-one child of the parent request.
        probs = disease.get("probs_by_label")
        db.add(
            PredictionDiseaseResult(
                prediction_request_id=record.id,
                label=disease.get("label", -1),
                label_name=disease.get("label_name"),
                probs_json=json.dumps(probs, ensure_ascii=True) if isinstance(probs, dict) else None,
            )
        )

    db.commit()
    # Refresh returns DB-populated fields such as autoincrement ids and
    # timestamps to callers that immediately serialize the record.
    db.refresh(record)
    return record


def get_prediction_logs(*, db: Session, limit: int = 50, offset: int = 0) -> list[PredictionRequest]:
    """Load prediction logs with related model, disease, and image records.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of parent prediction records to return.
        offset: Number of most-recent records to skip.

    Returns:
        List of prediction records ordered newest first, with child records
        eager-loaded for response serialization.
    """
    # Eager loading avoids N+1 queries when the logs endpoint serializes nested
    # image links, ensemble outputs, and disease subtype records.
    stmt = (
        select(PredictionRequest)
        .options(selectinload(PredictionRequest.image_links))
        .options(selectinload(PredictionRequest.binary_model_results))
        .options(selectinload(PredictionRequest.disease_result))
        .order_by(PredictionRequest.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(stmt).all())
