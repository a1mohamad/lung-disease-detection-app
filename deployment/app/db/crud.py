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
    db.flush()

    image_links = PredictionImageLink(
        prediction_request_id=record.id,
        source_url=response.get("source_url"),
        mask_url=response.get("mask_url"),
        roi_url=response.get("roi_url"),
        overlay_url=response.get("overlay_url"),
    )
    db.add(image_links)

    models_results = response.get("models_results")
    if isinstance(models_results, dict):
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
    db.refresh(record)
    return record


def get_prediction_logs(*, db: Session, limit: int = 50, offset: int = 0) -> list[PredictionRequest]:
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
