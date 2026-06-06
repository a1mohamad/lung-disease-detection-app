from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
import yaml
from keras.models import load_model


DEPLOYMENT_DIR = Path(__file__).resolve().parents[2]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from app.utils.metrics import dice_coefficient  # noqa: E402


@dataclass(frozen=True)
class ExportSpec:
    name: str
    model_dir: Path
    custom_objects: dict | None = None


MODEL_SPECS = (
    ExportSpec("binary_densenet", DEPLOYMENT_DIR / "saved_models/healthy_unhealthy/densenet"),
    ExportSpec("binary_efficientnet", DEPLOYMENT_DIR / "saved_models/healthy_unhealthy/efficientnet"),
    ExportSpec("binary_inception", DEPLOYMENT_DIR / "saved_models/healthy_unhealthy/inception"),
    ExportSpec("binary_mobilenet", DEPLOYMENT_DIR / "saved_models/healthy_unhealthy/mobilenet"),
    ExportSpec("diseases_densenet", DEPLOYMENT_DIR / "saved_models/diseases/densenet"),
    ExportSpec(
        "segmentation_unet_xception",
        DEPLOYMENT_DIR / "saved_models/segmentation/unet_xception",
        {"dice_coefficient": dice_coefficient},
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deployment Keras models to ONNX.")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--only", choices=[spec.name for spec in MODEL_SPECS], default=None)
    args = parser.parse_args()

    try:
        import tf2onnx
    except ImportError as exc:
        raise SystemExit("tf2onnx is not installed. Install deployment/requirements.txt first.") from exc

    specs = [spec for spec in MODEL_SPECS if args.only in (None, spec.name)]
    for spec in specs:
        export_model(spec, tf2onnx=tf2onnx, opset=args.opset)


def export_model(spec: ExportSpec, *, tf2onnx, opset: int) -> None:
    metadata = load_metadata(spec.model_dir)
    keras_path = spec.model_dir / metadata["model"]["path"]
    onnx_path = spec.model_dir / f"{keras_path.stem}.onnx"

    print(f"[export] {spec.name}: {keras_path.name} -> {onnx_path.name}")
    model = load_model(str(keras_path), custom_objects=spec.custom_objects)
    input_shape = infer_input_shape(model=model, metadata=metadata)
    input_signature = [
        tf.TensorSpec(input_shape, tf.float32, name="input"),
    ]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=str(onnx_path),
    )
    print(f"[done] {onnx_path}")


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.yaml"
    with metadata_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_input_shape(*, model, metadata: dict) -> list[int | None]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if len(input_shape) == 4 and all(dim is not None for dim in input_shape[1:]):
        return [None, int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]

    height, width = metadata.get("inference", {}).get("input_size", [256, 256])
    channels = metadata.get("inference", {}).get("channels", 3)
    return [None, int(height), int(width), int(channels)]


if __name__ == "__main__":
    main()
