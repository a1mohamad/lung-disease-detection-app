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
    def __init__(self, model_dir: Path) -> None:
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
        self.model = OnnxModelSession(self.model_path)

        classes = metadata.get("output", {}).get("classes", None)
        self.class_map = self._normalize_class_map(classes)
        if not self.class_map:
            self.class_map = self._load_json_map(AppConfig.DISEASES_JSON)

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

    def predict(self, roi_img) -> Dict[str, Any]:
        try:
            roi_img = ensure_batch(roi_img)
            roi_img = preprocess_torch_mode(roi_img)
            probs = np.squeeze(self.model.predict(roi_img), axis=0)
            label = int(np.argmax(probs, axis=-1))
            label_name = self.class_map.get(label)
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
