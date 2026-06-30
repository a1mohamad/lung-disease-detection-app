"""Keras binary classification wrappers and ensemble aggregation."""

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from app.configs.config import AppConfig
from app.preprocessing.pipeline import build_pipeline, run_pipeline
from app.utils.errors import ArtifactError, InferenceError, ModelError
from app.utils.metadata import load_metadata
from app.utils.model_loader import load_keras_model

class BinaryClassificationModel:
    """Run one Keras binary classifier with metadata-defined preprocessing.

    This wrapper mirrors the ONNX binary classifier but keeps the original Keras
    runtime available for local development, registry-backed experiments, and
    parity checks. The model threshold, preprocessing steps, and class mapping
    are loaded from metadata so the serving behavior follows the exported
    artifact rather than hard-coded assumptions.
    """

    def __init__(self, model_dir: Path, model_name: Optional[str] = None) -> None:
        """Load metadata, preprocessing steps, labels, and the Keras model.

        Args:
            model_dir: Directory containing ``metadata.yaml`` and the model file.
            model_name: Optional MLflow registered model name. When omitted, the
                name is inferred from the configured model directory.

        Raises:
            ArtifactError: If metadata lacks required model, threshold, or
            preprocessing fields.
            ModelError: If the Keras model cannot be loaded.
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
        try:
            registry_name = model_name
            if registry_name is None:
                registry_name = AppConfig.MLFLOW_MODEL_NAME_DENSENET_BINARY
                if "efficientnet" in str(model_dir).lower():
                    registry_name = AppConfig.MLFLOW_MODEL_NAME_EFFICIENTNET_BINARY
                elif "inception" in str(model_dir).lower():
                    registry_name = AppConfig.MLFLOW_MODEL_NAME_INCEPTION_BINARY
                elif "mobilenet" in str(model_dir).lower():
                    registry_name = AppConfig.MLFLOW_MODEL_NAME_MOBILENET_BINARY
            self.model = load_keras_model(
                model_dir=model_dir,
                model_rel_path=model_rel_path,
                model_name=registry_name,
            )
        except (OSError, ValueError) as exc:
            raise ModelError(
                "MODEL_LOAD_FAILED",
                "Failed to load classification model.",
                {"path": str(self.model_path)},
            ) from exc

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
            # Fallback JSON is retained for older saved models whose metadata
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


    def predict(self, img: tf.Tensor, mask: tf.Tensor) -> Tuple[float, int, Optional[str]]:
        """Predict one healthy/unhealthy label from an image and lung mask.

        Args:
            img: Image tensor supplied by the API preprocessing layer.
            mask: Lung segmentation mask used by metadata-driven transforms.

        Returns:
            Tuple of ``(unhealthy_probability, numeric_label, label_name)``.

        Raises:
            InferenceError: If preprocessing or Keras inference fails.
        """
        try:
            img = run_pipeline(img, mask, self.pipeline)
            prob = self.model.predict(img, verbose=0)[0].item()
            label = int(prob >= self.threshold)
            label_name = self.class_map.get(label)
            return prob, label, label_name
        except Exception as exc:
            raise InferenceError(
                "CLASSIFICATION_FAILED",
                "Classification prediction failed.",
                {"error": str(exc)},
            ) from exc


class EnsembleBinaryClassifier:
    """Aggregate multiple Keras binary classifiers into one screening result.

    The Keras ensemble is kept behaviorally aligned with the ONNX ensemble:
    each backbone predicts an unhealthy probability and the final decision is
    the arithmetic mean thresholded at 0.5.
    """

    def __init__(
        self, 
        models: Dict[str, BinaryClassificationModel],
    ) -> None:
        """Store participating models and the shared class map.

        Args:
            models: Mapping from ensemble member name to classifier wrapper.
        """
        self.models = models

        # Use the first model's class map as the shared ensemble display map.
        # All binary models are released with the same healthy/unhealthy labels.
        first_model = next(iter(models.values()))
        self.class_map = first_model.class_map

    def predict(
        self,
        img: tf.Tensor,
        mask: tf.Tensor,
        return_all: bool = True,
    ) -> Any:
        """Predict all binary models and return the averaged ensemble result.

        Args:
            img: Input image tensor.
            mask: Lung mask shared by all participating classifiers.
            return_all: Include each model's probability and label when true.

        Returns:
            Final ensemble probability, class label, label name, and optional
            per-model detail dictionary.
        """
        per_model: Dict[str, Dict[str, Any]] = {}
        for name, model in self.models.items():
            prob, label, label_name = model.predict(img, mask)
            probs_by_label = {
                "healthy": 1.0 - prob,
                "unhealthy": prob,
            }
            per_model[name] = {
                "prob": prob, 
                "probs_by_label": probs_by_label,
                "label": label,
                "label_name": label_name
            }

        all_probs = [v["prob"] for v in per_model.values()]

        # Keep the Keras and ONNX runtime paths explainable and equivalent.
        final_prob = sum(all_probs) / len(all_probs)
        final_label = int(final_prob >= 0.5)

        final_label_name = self.class_map.get(final_label)

        final_probs_by_label = {
            "healthy": 1.0 - final_prob,
            "unhealthy": final_prob,
        }

        if return_all:
            return {
                "final_prob": final_prob,
                "final_probs_by_label": final_probs_by_label,
                "final_label": final_label,
                "final_label_name": final_label_name,
                "models_results": per_model
            }
        return {
            "final_prob": final_prob,
            "final_probs_by_label": final_probs_by_label,
            "final_label": final_label,
            "final_label_name": final_label_name
        }
