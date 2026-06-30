"""ONNX segmentation wrapper for lung mask prediction."""

from pathlib import Path

import numpy as np

from app.configs.config import AppConfig
from app.preprocessing.transforms import normalize_image
from app.utils.errors import ArtifactError, InferenceError
from app.utils.metadata import load_metadata
from app.utils.onnx_loader import OnnxModelSession, get_onnx_model_path


class SegmentationOnnxModel:
    """ONNX-backed segmentation model that produces binary lung masks.

    The segmentation output is used twice: as a visible artifact for review and
    as a preprocessing signal for ROI cropping and masked classification.
    """

    def __init__(self, model_dir: Path):
        """Load segmentation metadata, threshold, normalization flag, and session.

        Args:
            model_dir: Directory containing the segmentation metadata and ONNX
                artifact.

        Raises:
            ArtifactError: If metadata lacks the model path or threshold.
            ModelError: If ONNX Runtime cannot load the artifact.
        """
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
        # Threshold and normalization flags are read from metadata so exported
        # ONNX artifacts stay aligned with their original Keras training setup.
        self.threshold = post_cfg["threshold"]
        self.is_normalized = inference_cfg.get("normalize", True)
        # The shared session wrapper hides provider selection and gives callers
        # a simple NumPy-in/NumPy-out prediction interface.
        self.model = OnnxModelSession(self.model_path)

    def predict_mask(self, img) -> np.ndarray:
        """Predict and threshold a lung mask for one image batch.

        Args:
            img: Image batch shaped for the segmentation model.

        Returns:
            Binary float32 mask with the batch dimension removed.

        Raises:
            InferenceError: If normalization or ONNX inference fails.
        """
        try:
            if not self.is_normalized:
                # Some segmentation artifacts were exported with raw image
                # expectations; normalize only when metadata says serving input
                # has not already been scaled.
                img = normalize_image(img, mode="imagenet")
            preds = self.model.predict(img)
            # Convert probability logits/scores into a binary mask using the
            # model-specific threshold recorded during training.
            pred_mask = (preds > self.threshold).astype(np.float32)
            return np.squeeze(pred_mask, axis=0)
        except Exception as exc:
            raise InferenceError(
                "SEGMENTATION_FAILED",
                "ONNX segmentation prediction failed.",
                {"error": str(exc), "path": str(self.model_path)},
            ) from exc
