"""ONNX binary classification wrappers and ensemble aggregation."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.configs.config import AppConfig
from app.preprocessing.pipeline import build_pipeline, run_pipeline
from app.utils.errors import ArtifactError, InferenceError
from app.utils.metadata import load_metadata
from app.utils.onnx_loader import OnnxModelSession, get_onnx_model_path


class BinaryClassificationOnnxModel:
    """Run one ONNX binary classifier with metadata-defined preprocessing.

    Each classifier owns its own preprocessing pipeline, threshold, ONNX
    session, and class mapping. Keeping these values in metadata makes the
    deployed artifact self-describing and prevents the API from hard-coding
    training-time assumptions.
    """

    def __init__(self, model_dir: Path, model_name: Optional[str] = None) -> None:
        """Load model metadata, preprocessing steps, labels, and ONNX session.

        Args:
            model_dir: Directory containing ``metadata.yaml`` and the ONNX file.
            model_name: Optional registered model name retained for parity with
                Keras wrappers and future logging extensions.

        Raises:
            ArtifactError: If required metadata fields are missing or invalid.
            ModelError: If the ONNX artifact cannot be resolved or loaded.
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
        if "threshold" not in inference_cfg:
            raise ArtifactError(
                "THRESHOLD_MISSING",
                "Metadata is missing classification threshold.",
                {"model_dir": str(model_dir)},
            )
        self.threshold = inference_cfg["threshold"]

        preprocessing_steps = metadata.get("preprocessing")
        if preprocessing_steps is None:
            raise ArtifactError(
                "PREPROCESSING_MISSING",
                "Metadata is missing preprocessing config.",
                {"model_dir": str(model_dir)},
            )
        self.pipeline = build_pipeline(preprocessing_steps)
        self.model = OnnxModelSession(self.model_path)

        classes = metadata.get("output", {}).get("classes", None)
        self.class_map = self._normalize_class_map(classes)
        if not self.class_map:
            self.class_map = self._load_json_map(AppConfig.CLASSIFICATION_JSON)

    def _normalize_class_map(self, classes: Optional[Dict[Any, Any]]) -> Dict[int, str]:
        """Convert metadata class labels into the runtime lookup format.

        Args:
            classes: Class mapping parsed from model metadata. YAML may load
                keys as strings, while predictions use integer labels.

        Returns:
            Dictionary keyed by integer class id with display labels.
        """
        if not classes:
            return {}
        return {int(k): v for k, v in classes.items()}

    def _load_json_map(self, path: Path) -> Dict[int, str]:
        """Load a fallback class mapping from a legacy JSON asset.

        Args:
            path: Path to the JSON class-map file.

        Returns:
            Dictionary keyed by integer class id. Missing files return an empty
            mapping so metadata-only artifacts continue normally.

        Raises:
            ArtifactError: If the file exists but is not valid JSON.
        """
        if not path.exists():
            return {}
        try:
            # Fallback JSON is retained for older ONNX exports whose metadata
            # did not yet include output class labels.
            with open(path, "r") as f:
                data = json.load(f)
        except JSONDecodeError as exc:
            raise ArtifactError(
                "CLASS_MAP_INVALID",
                "Class mapping file is not valid JSON.",
                {"path": str(path)},
            ) from exc

        return {int(k): v for k, v in data.items()}

    def predict(self, img, mask) -> Tuple[float, int, Optional[str]]:
        """Predict one healthy/unhealthy label from an image and lung mask.

        Args:
            img: Image batch supplied by the API preprocessing layer.
            mask: Lung segmentation mask used by metadata-driven transforms.

        Returns:
            Tuple of ``(unhealthy_probability, numeric_label, label_name)``.

        Raises:
            InferenceError: If preprocessing or ONNX inference fails.
        """
        try:
            img = run_pipeline(img, mask, self.pipeline)
            preds = self.model.predict(img)
            prob = float(np.squeeze(preds))
            label = int(prob >= self.threshold)
            label_name = self.class_map.get(label)
            return prob, label, label_name
        except Exception as exc:
            raise InferenceError(
                "CLASSIFICATION_FAILED",
                "ONNX classification prediction failed.",
                {"error": str(exc), "path": str(self.model_path)},
            ) from exc


class EnsembleBinaryOnnxClassifier:
    """Aggregate multiple ONNX binary classifiers into one screening result.

    The ensemble averages the unhealthy probability emitted by each model. This
    keeps the decision rule transparent and easy to audit while still allowing
    different CNN backbones to contribute to the final screening score.
    """

    def __init__(
        self,
        models: Dict[str, BinaryClassificationOnnxModel],
    ) -> None:
        """Store participating models and the shared class map.

        Args:
            models: Mapping from ensemble member name to ONNX classifier wrapper.
        """
        self.models = models

        first_model = next(iter(models.values()))
        self.class_map = first_model.class_map

    def predict(
        self,
        img,
        mask,
        return_all: bool = True,
    ) -> Any:
        """Predict all binary models and return the averaged ensemble result.

        Args:
            img: Input image batch.
            mask: Lung mask shared by all binary classifiers.
            return_all: Include each model's score and label when true.

        Returns:
            A dictionary with final probability, final label, label name, and
            optional per-model details.
        """
        per_model: Dict[str, Dict[str, Any]] = {}
        for name, model in self.models.items():
            prob, label, label_name = model.predict(img, mask)
            per_model[name] = {
                "prob": prob,
                "probs_by_label": {
                    "healthy": 1.0 - prob,
                    "unhealthy": prob,
                },
                "label": label,
                "label_name": label_name,
            }

        all_probs = [v["prob"] for v in per_model.values()]

        # A simple probability mean is intentionally used here: it is stable,
        # explainable, and matches the artifact metadata used during release.
        final_prob = sum(all_probs) / len(all_probs)
        final_label = int(final_prob >= 0.5)
        final_label_name = self.class_map.get(final_label)

        result = {
            "final_prob": final_prob,
            "final_probs_by_label": {
                "healthy": 1.0 - final_prob,
                "unhealthy": final_prob,
            },
            "final_label": final_label,
            "final_label_name": final_label_name,
        }
        if return_all:
            result["models_results"] = per_model
        return result
