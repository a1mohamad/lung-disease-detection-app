"""MLflow model signature helpers for Keras models."""

from __future__ import annotations

import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec


def _to_tensor_specs(tensors, prefix: str) -> list[TensorSpec]:
    """Convert Keras tensors to MLflow tensor specs.

    Args:
        tensors: Keras tensor or list of tensors.
        prefix: Fallback name prefix for unnamed tensors.

    Returns:
        List of MLflow tensor specifications.
    """
    if tensors is None:
        return []
    raw = tensors if isinstance(tensors, list) else [tensors]
    specs: list[TensorSpec] = []
    for idx, tensor in enumerate(raw):
        if tensor is None:
            continue
        # MLflow signatures use -1 for dynamic dimensions, while Keras exposes
        # them as None.
        shape = tuple(-1 if dim is None else int(dim) for dim in tensor.shape)
        dtype_name = getattr(getattr(tensor, "dtype", None), "name", str(getattr(tensor, "dtype", "float32")))
        dtype = np.dtype(dtype_name)
        name = getattr(tensor, "name", f"{prefix}_{idx}")
        specs.append(TensorSpec(type=dtype, shape=shape, name=name))
    return specs


def build_keras_model_signature(model) -> ModelSignature | None:
    """Build an MLflow signature directly from Keras model tensor specs.

    Args:
        model: Keras model with input and output tensors.

    Returns:
        MLflow model signature, or ``None`` when tensor metadata is unavailable.
    """
    try:
        input_specs = _to_tensor_specs(getattr(model, "inputs", None), "input")
        output_specs = _to_tensor_specs(getattr(model, "outputs", None), "output")
        if not input_specs or not output_specs:
            return None
        return ModelSignature(inputs=Schema(input_specs), outputs=Schema(output_specs))
    except Exception:
        # Signature logging should improve registry quality, but it should not
        # block backfill/retraining when a legacy model exposes unusual tensors.
        return None
