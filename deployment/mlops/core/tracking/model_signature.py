from __future__ import annotations

import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec


def _to_tensor_specs(tensors, prefix: str) -> list[TensorSpec]:
    if tensors is None:
        return []
    raw = tensors if isinstance(tensors, list) else [tensors]
    specs: list[TensorSpec] = []
    for idx, tensor in enumerate(raw):
        if tensor is None:
            continue
        shape = tuple(-1 if dim is None else int(dim) for dim in tensor.shape)
        dtype_name = getattr(getattr(tensor, "dtype", None), "name", str(getattr(tensor, "dtype", "float32")))
        dtype = np.dtype(dtype_name)
        name = getattr(tensor, "name", f"{prefix}_{idx}")
        specs.append(TensorSpec(type=dtype, shape=shape, name=name))
    return specs


def build_keras_model_signature(model) -> ModelSignature | None:
    """Build an MLflow signature directly from keras model tensor specs."""
    try:
        input_specs = _to_tensor_specs(getattr(model, "inputs", None), "input")
        output_specs = _to_tensor_specs(getattr(model, "outputs", None), "output")
        if not input_specs or not output_specs:
            return None
        return ModelSignature(inputs=Schema(input_specs), outputs=Schema(output_specs))
    except Exception:
        return None

