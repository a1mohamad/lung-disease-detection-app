"""ONNX disease subtype classifier for scans predicted as unhealthy."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.configs.config import AppConfig
from app.preprocessing.model_preprocessing import preprocess_torch_mode
from app.preprocessing.transforms import ensure_batch
from app.utils.errors import ArtifactError, InferenceError
from app.utils.metadata import load_metadata
from app.utils.onnx_loader import OnnxModelSession, get_onnx_model_path


class DiseasesOnnxClassifier:
    """ONNX-backed multiclass classifier for unhealthy scan subtyping.

    The classifier is intentionally separate from the binary ensemble. It is
    only called after the screening stage marks a scan as unhealthy, which keeps
    healthy predictions from receiving misleading disease labels.
    """

    def __init__(self, model_dir: Path) -> None:
        """Load disease classifier metadata, class labels, and ONNX session.

        Args:
            model_dir: Directory containing disease model metadata and artifact.

        Raises:
            ArtifactError: If required metadata or class-map files are invalid.
            ModelError: If the ONNX session cannot be created.
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
        # The ONNX session wrapper standardizes runtime errors and provider
        # selection for all exported inference artifacts.
        self.model = OnnxModelSession(self.model_path)

        classes = metadata.get("output", {}).get("classes", None)
        # Prefer metadata labels because they travel with the promoted artifact;
        # fall back to the legacy JSON map for older saved models.
        self.class_map = self._normalize_class_map(classes)
        if not self.class_map:
            self.class_map = self._load_json_map(AppConfig.DISEASES_JSON)

    def _normalize_class_map(self, classes: Optional[Dict[Any, Any]]) -> Dict[int, str]:
        """Convert metadata class labels into the runtime lookup format.

        Args:
            classes: Class mapping parsed from model metadata. YAML can parse
                keys as strings or integers depending on the file.

        Returns:
            Dictionary keyed by integer class id with human-readable labels.
        """
        if not classes:
            return {}
        return {int(k): v for k, v in classes.items()}

    def _load_json_map(self, path: Path) -> Dict[int, str]:
        """Load a fallback disease class mapping from a legacy JSON asset.

        Args:
            path: Path to the JSON class map.

        Returns:
            Dictionary keyed by integer class id. Missing files return an empty
            mapping so metadata-only deployments continue normally.

        Raises:
            ArtifactError: If the file exists but is not valid JSON.
        """
        if not path.exists():
            return {}
        try:
            # The class map is small and operational, so strict JSON parsing is
            # preferable to silently accepting malformed labels.
            with open(path, "r") as f:
                data = json.load(f)
        except JSONDecodeError as exc:
            raise ArtifactError(
                "CLASS_MAP_INVALID",
                "Class mapping file is not valid JSON.",
                {"path": str(path)},
            ) from exc

        return {int(k): v for k, v in data.items()}

    def predict(self, roi_img) -> Dict[str, Any]:
        """Predict the most likely disease subtype for a lung ROI.

        Args:
            roi_img: Cropped lung-region image produced by segmentation.

        Returns:
            Dictionary with class probabilities, winning numeric label, and
            human-readable label name.

        Raises:
            InferenceError: If preprocessing or ONNX inference fails.
        """
        try:
            # Disease models are DenseNet-family artifacts trained with
            # ImageNet/Torch normalization over a batched ROI tensor.
            roi_img = ensure_batch(roi_img)
            roi_img = preprocess_torch_mode(roi_img)
            probs = np.squeeze(self.model.predict(roi_img), axis=0)
            label = int(np.argmax(probs, axis=-1))
            label_name = self.class_map.get(label)
            # Probability maps are keyed by display label to keep API and log
            # payloads readable for clinicians and reviewers.
            probs_by_label = {
                self.class_map.get(i, str(i)): float(probs[i])
                for i in range(len(probs))
            }

            return {
                "probs_by_label": probs_by_label,
                "label": label,
                "label_name": label_name,
            }
        except Exception as exc:
            raise InferenceError(
                "DISEASES_PREDICT_FAILED",
                "ONNX diseases prediction failed.",
                {"error": str(exc), "path": str(self.model_path)},
            ) from exc
