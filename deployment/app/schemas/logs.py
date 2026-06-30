"""Pydantic schemas for prediction log responses."""

from datetime import datetime

from pydantic import BaseModel, Field


# These schemas translate normalized SQL rows back into the nested shape that
# operators expect from the protected prediction-log API.
class PredictionImageLinksItem(BaseModel):
    """Generated artifact links and storage paths attached to a log item.

    The API returns both URLs and storage paths so operators can inspect images
    through the frontend while still retaining stable object identifiers for
    audits or downstream review queues.
    """

    source_url: str | None = None
    mask_url: str | None = None
    roi_url: str | None = None
    overlay_url: str | None = None
    source_path: str | None = None
    mask_path: str | None = None
    roi_path: str | None = None
    overlay_path: str | None = None


class BinaryModelResultItem(BaseModel):
    """Logged output from one binary ensemble member.

    Attributes:
        model_name: Ensemble member identifier.
        label: Thresholded numeric prediction.
        label_name: Human-readable label name.
        prob: Unhealthy probability from this model.
        probs_by_label: Optional probability map serialized from the log row.
    """

    model_name: str
    label: int
    label_name: str | None = None
    prob: float
    probs_by_label: dict[str, float] | None = None


class DiseaseResultItem(BaseModel):
    """Logged disease subtype output.

    Attributes:
        label: Numeric disease class.
        label_name: Human-readable disease class name.
        probs_by_label: Optional class-probability map.
    """

    label: int
    label_name: str | None = None
    probs_by_label: dict[str, float] | None = None


class PredictionLogItem(BaseModel):
    """One normalized prediction log record.

    The shape mirrors the API prediction response plus operational metadata
    such as request id, source type, errors, timestamps, and child records.
    """

    id: int
    request_id: str
    input_type: str
    final_label: int
    final_label_name: str | None = None
    final_prob: float
    final_probs_by_label: dict[str, float] | None = None
    # Default factories avoid mutable class-level lists while giving empty logs
    # a clean JSON shape when per-model rows are absent.
    binary_model_results: list[BinaryModelResultItem] = Field(default_factory=list)
    disease_result: DiseaseResultItem | None = None
    error_code: str | None = None
    error_message: str | None = None
    image_links: PredictionImageLinksItem | None = None
    created_at: datetime


class PredictionLogsResponse(BaseModel):
    """Paginated response for the protected logs endpoint.

    Attributes:
        items: Current page of prediction log records.
        limit: Page size applied by the API.
        offset: Number of newest records skipped.
    """

    items: list[PredictionLogItem]
    limit: int
    offset: int
