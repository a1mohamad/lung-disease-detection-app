from datetime import datetime

from pydantic import BaseModel, Field


class PredictionImageLinksItem(BaseModel):
    source_url: str | None = None
    mask_url: str | None = None
    roi_url: str | None = None
    overlay_url: str | None = None


class BinaryModelResultItem(BaseModel):
    model_name: str
    label: int
    label_name: str | None = None
    prob: float
    probs_by_label: dict[str, float] | None = None


class DiseaseResultItem(BaseModel):
    label: int
    label_name: str | None = None
    probs_by_label: dict[str, float] | None = None


class PredictionLogItem(BaseModel):
    id: int
    request_id: str
    input_type: str
    final_label: int
    final_label_name: str | None = None
    final_prob: float
    final_probs_by_label: dict[str, float] | None = None
    binary_model_results: list[BinaryModelResultItem] = Field(default_factory=list)
    disease_result: DiseaseResultItem | None = None
    error_code: str | None = None
    error_message: str | None = None
    image_links: PredictionImageLinksItem | None = None
    created_at: datetime


class PredictionLogsResponse(BaseModel):
    items: list[PredictionLogItem]
    limit: int
    offset: int
