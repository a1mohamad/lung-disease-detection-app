from pydantic import BaseModel
from typing import Dict, Optional

class ModelResult(BaseModel):
    prob: float
    probs_by_label: Dict[str, float]
    label: int
    label_name: Optional[str] = None

class DiseaseResult(BaseModel):
    probs_by_label: Dict[str, float]
    label: int
    label_name: Optional[str] = None

class PredictResponse(BaseModel):
    final_prob: float
    final_probs_by_label: Dict[str, float]
    final_label: int
    final_label_name: Optional[str] = None
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
