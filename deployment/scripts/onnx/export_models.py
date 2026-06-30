"""Export deployment Keras models to ONNX artifacts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from keras.models import load_model


DEPLOYMENT_DIR = Path(__file__).resolve().parents[2]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from app.configs.constants import ONNX_EXPORT_DEFAULT_OPSET  # noqa: E402
from app.utils.metadata import load_metadata  # noqa: E402
from app.utils.metrics import dice_coefficient  # noqa: E402
from app.utils.onnx_loader import get_onnx_model_path  # noqa: E402


@dataclass(frozen=True)
class ExportSpec:
    """Local model export target and optional Keras custom objects.

    Attributes:
        name: CLI-friendly export target name.
        model_dir: Directory containing metadata and the source Keras artifact.
        custom_objects: Optional custom Keras objects needed to load the model.
    """

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
    """Parse CLI arguments and export selected models to ONNX.

    Raises:
        SystemExit: If ``tf2onnx`` is not installed.
    """
    parser = argparse.ArgumentParser(description="Export deployment Keras models to ONNX.")
    parser.add_argument("--opset", type=int, default=ONNX_EXPORT_DEFAULT_OPSET)
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
    """Export one Keras model to the ONNX path declared by metadata.

    Args:
        spec: Export target specification.
        tf2onnx: Imported ``tf2onnx`` module.
        opset: ONNX opset version.
    """
    metadata = load_metadata(spec.model_dir)
    keras_path = spec.model_dir / metadata["model"]["path"]
    onnx_path = get_onnx_model_path(spec.model_dir, metadata)

    print(f"[export] {spec.name}: {keras_path.name} -> {onnx_path.name}")
    model = load_model(str(keras_path), custom_objects=spec.custom_objects)
    input_shape = infer_input_shape(model=model, metadata=metadata)
    # Use the metadata/model-derived input signature so exported ONNX models
    # receive the same tensor shape expected by the API runtime.
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


def infer_input_shape(*, model, metadata: dict) -> list[int | None]:
    """Infer an export input shape from the model or metadata fallback.

    Args:
        model: Loaded Keras model.
        metadata: Parsed model metadata.

    Returns:
        Input shape list using ``None`` for the dynamic batch dimension.
    """
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
