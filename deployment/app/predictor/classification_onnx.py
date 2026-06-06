import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from app.configs.config import AppConfig
from app.preprocessing.pipeline import build_pipeline, run_pipeline
from app.utils.errors import ArtifactError, InferenceError
from app.utils.metadata import load_metadata
from app.utils.onnx_loader import OnnxModelSession, get_onnx_model_path


class BinaryClassificationOnnxModel:
    def __init__(self, model_dir: Path, model_name: Optional[str] = None) -> None:
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
        if not classes:
            return {}
        return {int(k): v for k, v in classes.items()}

    def _load_json_map(self, path: Path) -> Dict[int, str]:
        if not path.exists():
            return {}
        try:
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
        try:
            img = run_pipeline(img, mask, self.pipeline)
            preds = self.model.predict(img.numpy())
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
    def __init__(
        self,
        models: Dict[str, BinaryClassificationOnnxModel],
        vote_threshold: int = 2,
    ) -> None:
        self.models = models
        self.vote_threshold = vote_threshold

        first_model = next(iter(models.values()))
        self.class_map = first_model.class_map

    def predict(
        self,
        img: tf.Tensor,
        mask: tf.Tensor,
        return_all: bool = True,
    ) -> Any:
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

        all_labels = [v["label"] for v in per_model.values()]
        all_probs = [v["prob"] for v in per_model.values()]

        final_label = int(sum(all_labels) > self.vote_threshold)
        final_prob = sum(all_probs) / len(all_probs)
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
