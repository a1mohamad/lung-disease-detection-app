"""ONNX Runtime session wrapper and model path resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.utils.errors import ModelError

logger = logging.getLogger(__name__)


class OnnxModelSession:
    """Small wrapper around an ONNX Runtime CPU inference session.

    The wrapper centralizes ONNX Runtime import errors, artifact validation,
    provider selection, and input/output name discovery. Model-specific classes
    can then call ``predict`` without duplicating session boilerplate.
    """

    def __init__(self, model_path: Path) -> None:
        """Load an ONNX model and cache its input and output names.

        Args:
            model_path: Filesystem path to the ONNX artifact.

        Raises:
            ModelError: If ONNX Runtime is missing, the file does not exist, or
            the session cannot be created.
        """
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
        """Run inference and return the first output tensor as a NumPy array.

        Args:
            input_array: Input tensor convertible to float32.

        Returns:
            First output produced by the ONNX session.
        """
        array = np.asarray(input_array, dtype=np.float32)
        outputs = self.session.run(self.output_names[:1], {self.input_name: array})
        return outputs[0]


def get_onnx_model_path(model_dir: Path, metadata: dict) -> Path:
    """Resolve the ONNX artifact path declared by metadata or Keras fallback.

    Args:
        model_dir: Directory containing metadata and model artifacts.
        metadata: Parsed model metadata.

    Returns:
        Path to the ONNX artifact expected by the runtime.

    Raises:
        ModelError: If metadata does not contain enough model path information.
    """
    model_cfg = metadata.get("model", {})
    onnx_rel_path = model_cfg.get("onnx_path")
    # Prefer explicit ONNX metadata from the release pipeline. The fallback keeps
    # older metadata compatible by deriving "model.onnx" from "model.keras".
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
