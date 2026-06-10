from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.errors import ModelError

logger = logging.getLogger(__name__)


class OnnxModelSession:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelError(
                "ONNX_RUNTIME_NOT_INSTALLED",
                "MODEL_RUNTIME=onnx but onnxruntime is not installed.",
            ) from exc

        if not model_path.exists():
            raise ModelError(
                "ONNX_MODEL_NOT_FOUND",
                "ONNX model file was not found.",
                {"path": str(model_path)},
            )

        try:
            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ModelError(
                "ONNX_MODEL_LOAD_FAILED",
                "Failed to load ONNX model.",
                {"path": str(model_path), "error": str(exc)},
            ) from exc

        self.model_path = model_path
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        logger.info("ONNX model loaded. path=%s input=%s", model_path, self.input_name)

    def predict(self, input_array: Any) -> np.ndarray:
        array = np.asarray(input_array, dtype=np.float32)
        outputs = self.session.run(self.output_names[:1], {self.input_name: array})
        return outputs[0]


def get_onnx_model_path(model_dir: Path, metadata: dict) -> Path:
    model_cfg = metadata.get("model", {})
    onnx_rel_path = model_cfg.get("onnx_path")
    if onnx_rel_path:
        return model_dir / onnx_rel_path

    keras_rel_path = model_cfg.get("path")
    if not keras_rel_path:
        raise ModelError(
            "MODEL_PATH_MISSING",
            "Metadata is missing model path.",
            {"model_dir": str(model_dir)},
        )

    return model_dir / f"{Path(keras_rel_path).stem}.onnx"
