from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from keras.models import load_model


DEPLOYMENT_DIR = Path(__file__).resolve().parents[2]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from app.utils.metrics import dice_coefficient  # noqa: E402
from export_models import MODEL_SPECS, infer_input_shape  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Keras and ONNX outputs.")
    parser.add_argument("--only", choices=[spec.name for spec in MODEL_SPECS], default=None)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
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
    metadata = load_metadata(spec.model_dir)
    keras_path = spec.model_dir / metadata["model"]["path"]
    onnx_path = spec.model_dir / f"{keras_path.stem}.onnx"
    if not onnx_path.exists():
        print(f"[missing] {spec.name}: {onnx_path}")
        return False

    keras_model = load_model(str(keras_path), custom_objects=spec.custom_objects)
    input_shape = infer_input_shape(model=keras_model, metadata=metadata)
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


def load_metadata(model_dir: Path) -> dict:
    with (model_dir / "metadata.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_sample(input_shape: list[int | None]) -> np.ndarray:
    shape = [1 if dim is None else dim for dim in input_shape]
    rng = np.random.default_rng(42)
    return rng.uniform(0, 255, size=shape).astype(np.float32)


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
