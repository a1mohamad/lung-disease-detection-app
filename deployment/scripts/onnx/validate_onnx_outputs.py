"""Validate numerical agreement between Keras and ONNX model outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model


DEPLOYMENT_DIR = Path(__file__).resolve().parents[2]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from app.configs.constants import (  # noqa: E402
    ONNX_VALIDATION_DEFAULT_ATOL,
    ONNX_VALIDATION_DEFAULT_RTOL,
    ONNX_VALIDATION_RANDOM_SEED,
)
from app.utils.metadata import load_metadata  # noqa: E402
from app.utils.metrics import dice_coefficient  # noqa: E402
from app.utils.onnx_loader import get_onnx_model_path  # noqa: E402
from export_models import MODEL_SPECS, infer_input_shape  # noqa: E402


def main() -> None:
    """Parse CLI arguments and validate selected ONNX exports.

    Raises:
        SystemExit: If ONNX Runtime is missing or any selected model fails
        numerical validation.
    """
    parser = argparse.ArgumentParser(description="Compare Keras and ONNX outputs.")
    parser.add_argument("--only", choices=[spec.name for spec in MODEL_SPECS], default=None)
    parser.add_argument("--rtol", type=float, default=ONNX_VALIDATION_DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=ONNX_VALIDATION_DEFAULT_ATOL)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is not installed. Install deployment/requirements.txt first.") from exc

    specs = [spec for spec in MODEL_SPECS if args.only in (None, spec.name)]
    failures = []
    for spec in specs:
        ok = validate_model(spec, ort=ort, rtol=args.rtol, atol=args.atol)
        if not ok:
            failures.append(spec.name)

    if failures:
        raise SystemExit(f"ONNX validation failed for: {', '.join(failures)}")


def validate_model(spec, *, ort, rtol: float, atol: float) -> bool:
    """Return whether one ONNX model matches its Keras source within tolerance.

    Args:
        spec: Export target specification.
        ort: Imported ONNX Runtime module.
        rtol: Relative tolerance for output comparison.
        atol: Absolute tolerance for output comparison.

    Returns:
        True when ONNX and Keras outputs are close within tolerance.
    """
    metadata = load_metadata(spec.model_dir)
    keras_path = spec.model_dir / metadata["model"]["path"]
    onnx_path = get_onnx_model_path(spec.model_dir, metadata)
    if not onnx_path.exists():
        print(f"[missing] {spec.name}: {onnx_path}")
        return False

    keras_model = load_model(str(keras_path), custom_objects=spec.custom_objects)
    input_shape = infer_input_shape(model=keras_model, metadata=metadata)
    # A deterministic random sample catches export mismatches while keeping the
    # validation script repeatable in CI and release jobs.
    sample = make_sample(input_shape)

    keras_output = keras_model.predict(sample, verbose=0)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: sample})[0]

    max_abs = float(np.max(np.abs(keras_output - onnx_output)))
    mean_abs = float(np.mean(np.abs(keras_output - onnx_output)))
    ok = np.allclose(keras_output, onnx_output, rtol=rtol, atol=atol)
    status = "ok" if ok else "fail"
    print(
        f"[{status}] {spec.name}: shape={onnx_output.shape} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
    )
    return ok


def make_sample(input_shape: list[int | None]) -> np.ndarray:
    """Create a deterministic random sample for model-output validation.

    Args:
        input_shape: Model input shape with optional dynamic dimensions.

    Returns:
        Float32 sample batch with dynamic dimensions replaced by ``1``.
    """
    shape = [1 if dim is None else dim for dim in input_shape]
    rng = np.random.default_rng(ONNX_VALIDATION_RANDOM_SEED)
    return rng.uniform(0, 255, size=shape).astype(np.float32)


if __name__ == "__main__":
    tf.random.set_seed(ONNX_VALIDATION_RANDOM_SEED)
    main()
