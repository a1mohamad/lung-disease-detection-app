import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional

from app.configs.config import AppConfig
from keras.applications.densenet import preprocess_input
from app.preprocessing.transforms import ensure_batch
import tensorflow as tf
from app.utils.metadata import load_metadata
from app.utils.errors import ArtifactError, InferenceError, ModelError
from app.utils.model_loader import load_keras_model


class DiseasesClassifier:
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
        self.model_path = model_dir / model_rel_path
        try:
            self.model = load_keras_model(
                model_dir=model_dir,
                model_rel_path=model_rel_path,
                model_name=AppConfig.MLFLOW_MODEL_NAME_DISEASES,
            )
        except (OSError, ValueError) as exc:
            raise ModelError(
                "MODEL_LOAD_FAILED",
                "Failed to load model.",
                {"path": str(self.model_path)},
            ) from exc
        self.preprocess = preprocess_input

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

    def predict(self, roi_img: tf.Tensor) -> Dict[str, Any]:
        try:
            roi_img = ensure_batch(roi_img)
            roi_img = self.preprocess(roi_img)
            probs = self.model.predict(roi_img, verbose=0)[0]
            label = int(tf.argmax(probs, axis=-1).numpy())
            label_name = self.class_map.get(label)
            probs_by_label = {
                self.class_map.get(i, str(i)): float(probs[i])
                for i in range(len(probs))
            }

            return {
                "probs_by_label": probs_by_label,
                "label": label,
                "label_name": label_name
            }
        except Exception as exc:
            raise InferenceError(
                "DISEASES_PREDICT_FAILED",
                "Diseases prediction failed.",
                {"error": str(exc)},
            ) from exc
        


