from pathlib import Path

import numpy as np

from app.configs.config import AppConfig
from app.preprocessing.transforms import normalize_image
from app.utils.errors import ArtifactError, InferenceError
from app.utils.metadata import load_metadata
from app.utils.onnx_loader import OnnxModelSession, get_onnx_model_path


class SegmentationOnnxModel:
    def __init__(self, model_dir: Path):
        metadata = load_metadata(model_dir)

        model_cfg = metadata.get("model", {})
        model_rel_path = model_cfg.get("path")
        if not model_rel_path:
            raise ArtifactError(
                "MODEL_PATH_MISSING",
                "Metadata is missing model path.",
                {"model_dir": str(model_dir)},
            )
        self.model_path = get_onnx_model_path(model_dir, metadata)

        inference_cfg = metadata.get("inference", {})
        post_cfg = inference_cfg.get("postprocessing", {})
        if "threshold" not in post_cfg:
            raise ArtifactError(
                "THRESHOLD_MISSING",
                "Metadata is missing segmentation threshold.",
                {"model_dir": str(model_dir)},
            )
        self.threshold = post_cfg["threshold"]
        self.is_normalized = inference_cfg.get("normalize", True)
        self.model = OnnxModelSession(self.model_path)

    def predict_mask(self, img) -> np.ndarray:
        try:
            if not self.is_normalized:
                img = normalize_image(img, mode="imagenet")
            preds = self.model.predict(img)
            pred_mask = (preds > self.threshold).astype(np.float32)
            return np.squeeze(pred_mask, axis=0)
        except Exception as exc:
            raise InferenceError(
                "SEGMENTATION_FAILED",
                "ONNX segmentation prediction failed.",
                {"error": str(exc), "path": str(self.model_path)},
            ) from exc
