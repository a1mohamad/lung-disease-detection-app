"""Pydantic response schemas for prediction endpoints."""

from pydantic import BaseModel
from typing import Dict, Optional


# Response models form the public API contract. They intentionally mirror the
# inference-service dictionary keys so FastAPI validation catches drift quickly.
class ModelResult(BaseModel):
    """Prediction details from one binary classifier.

    Attributes:
        prob: Unhealthy probability emitted by this model.
        probs_by_label: Human-readable probability map for healthy/unhealthy.
        label: Thresholded numeric label.
        label_name: Optional class name resolved from metadata.
    """

    prob: float
    probs_by_label: Dict[str, float]
    label: int
    label_name: Optional[str] = None

class DiseaseResult(BaseModel):
    """Disease subtype prediction details.

    Attributes:
        probs_by_label: Probability map keyed by disease label name.
        label: Numeric disease class selected by argmax.
        label_name: Human-readable disease class name.
    """

    probs_by_label: Dict[str, float]
    label: int
    label_name: Optional[str] = None

class PredictResponse(BaseModel):
    """Public prediction response returned by the inference API.

    Attributes:
        final_prob: Ensemble unhealthy probability.
        final_probs_by_label: Healthy/unhealthy probability map.
        final_label: Final numeric binary label.
        final_label_name: Human-readable binary label.
        models_results: Optional per-model ensemble details.
        disease: Optional disease subtype, present only for unhealthy scans.
        source_url: Public URL for the source artifact when available.
        mask_url: Public URL for the predicted mask artifact.
        roi_url: Public URL for the cropped ROI artifact.
        overlay_url: Public URL for the mask overlay artifact.
        source_path: Storage-relative source artifact path.
        mask_path: Storage-relative mask artifact path.
        roi_path: Storage-relative ROI artifact path.
        overlay_path: Storage-relative overlay artifact path.
    """

    final_prob: float
    final_probs_by_label: Dict[str, float]
    final_label: int
    final_label_name: Optional[str] = None
    # Detailed ensemble and subtype output is optional because simple clients
    # only need the final binary decision and artifact links.
    models_results: Optional[Dict[str, ModelResult]] = None
    disease: Optional[DiseaseResult] = None
    source_url: Optional[str] = None
    mask_url: Optional[str] = None
    roi_url: Optional[str] = None
    overlay_url: Optional[str] = None
    source_path: Optional[str] = None
    mask_path: Optional[str] = None
    roi_path: Optional[str] = None
    overlay_path: Optional[str] = None
