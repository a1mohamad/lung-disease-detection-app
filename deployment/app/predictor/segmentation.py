"""Keras segmentation wrapper for lung mask prediction."""

from pathlib import Path

import tensorflow as tf

from app.configs.config import AppConfig
from app.preprocessing.transforms import normalize_image
from app.utils.errors import ArtifactError, InferenceError, ModelError
from app.utils.metadata import load_metadata
from app.utils.metrics import dice_coefficient
from app.utils.model_loader import load_keras_model

class SegmentationModel:
    """Keras-backed segmentation model that produces binary lung masks.

    The segmentation model is the first model in the inference pipeline. Its
    mask becomes both a review artifact and an input to ROI/masked classifier
    preprocessing.
    """

    def __init__(self, model_dir: Path):
        """Load segmentation metadata, threshold, normalization flag, and model.

        Args:
            model_dir: Directory containing segmentation metadata and Keras
                artifact.

        Raises:
            ArtifactError: If required metadata fields are missing.
            ModelError: If the Keras segmentation model cannot be loaded.
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
        self.model_path = model_dir / model_rel_path

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

        try:
            self.model = load_keras_model(
                model_dir=model_dir,
                model_rel_path=model_rel_path,
                model_name=AppConfig.MLFLOW_MODEL_NAME_SEGMENTATION,
                custom_objects={"dice_coefficient": dice_coefficient},
            )
        except (OSError, ValueError) as exc:
            raise ModelError(
                "MODEL_LOAD_FAILED",
                "Failed to load segmentation model.",
                {"path": str(self.model_path)},
            ) from exc

    def predict_mask(self, img: tf.Tensor) -> tf.Tensor:
        """Predict and threshold a lung mask for one image batch.

        Args:
            img: Image batch shaped for the segmentation model.

        Returns:
            Binary float32 mask with the batch dimension removed.

        Raises:
            InferenceError: If normalization, prediction, or thresholding fails.
        """
        try:
            if not self.is_normalized:
                # Some exported segmentation models expect ImageNet-style input
                # even though the API image loader preserves raw [0, 255] values.
                img = normalize_image(img, mode="imagenet")
            preds = self.model(img, verbose=0)
            pred_mask = tf.cast(preds > self.threshold, tf.float32)
            pred_mask = tf.squeeze(pred_mask, axis=0)
            return pred_mask
        except Exception as exc:
            raise InferenceError(
                "SEGMENTATION_FAILED",
                "Segmentation prediction failed.",
                {"error": str(exc)},
            ) from exc
