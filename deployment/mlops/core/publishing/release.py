"""Model release staging, ONNX conversion, validation, and Hub publishing."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlops.config.model_specs import SAVED_MODELS_DIR
from mlops.config.settings import MLOpsSettings


@dataclass(frozen=True)
class ModelRelease:
    """Paths and repository target information for a staged model release.

    Attributes:
        root: Release root directory for the MLflow run.
        model_dir: Model-specific release directory.
        path_in_repo: Hugging Face repository path for upload.
        keras_path: Saved Keras artifact path.
        onnx_path: Exported ONNX artifact path.
    """

    root: Path
    model_dir: Path
    path_in_repo: str
    keras_path: Path
    onnx_path: Path


def stage_model_release(*, model, spec, metadata: dict, run_id: str) -> ModelRelease:
    """Save, convert, and validate artifacts for a promoted model candidate.

    A release contains the Keras model, ONNX export, metadata, and validation
    report in the same directory layout expected by the inference repository.
    ONNX validation happens before the release object is returned so promotion
    cannot publish an artifact that disagrees with the Keras source model.
    """
    try:
        # Import ONNX Runtime lazily so training/evaluation jobs can run in
        # lighter environments that do not perform release validation.
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX release dependencies are missing from the MLOps environment."
        ) from exc

    relative_model_dir = spec.model_dir.relative_to(SAVED_MODELS_DIR)
    release_root = MLOpsSettings.MODEL_RELEASE_ROOT / run_id
    model_dir = release_root / relative_model_dir
    if release_root.exists():
        # Release directories are immutable by run id; overwriting them would
        # make audit trails and model-registry links ambiguous.
        raise RuntimeError(f"Model release directory already exists: {release_root}")
    model_dir.mkdir(parents=True)

    model_config = metadata.get("model", {})
    keras_name = model_config.get("path")
    onnx_name = model_config.get("onnx_path")
    if not keras_name or not onnx_name:
        raise ValueError(f"Model metadata lacks Keras or ONNX path: {spec.metadata_path}")

    keras_path = model_dir / keras_name
    onnx_path = model_dir / onnx_name
    # The release directory mirrors the inference repository layout so it can be
    # copied or uploaded without a second packaging step.
    model.save(str(keras_path))
    shutil.copy2(spec.metadata_path, model_dir / "metadata.yaml")

    input_shape = _infer_input_shape(model, metadata)
    saved_model_dir = release_root / ".onnx_saved_model"
    # Keras exports a temporary SavedModel first because tf2onnx consumes that
    # format more reliably than a direct .keras conversion path.
    model.export(str(saved_model_dir))
    _convert_saved_model(saved_model_dir, onnx_path)
    # Promotion must prove that the exported ONNX graph matches the Keras model
    # before anything is logged as a releasable artifact.
    validation = _validate_onnx(model, onnx_path, input_shape, ort)
    # Remove temporary TensorFlow export output; only deployable artifacts and
    # release metadata should remain in the staged folder.
    shutil.rmtree(saved_model_dir)
    (model_dir / "release.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "registered_model_name": spec.registered_name,
                "onnx_validation": validation,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ModelRelease(
        root=release_root,
        model_dir=model_dir,
        path_in_repo=relative_model_dir.as_posix(),
        keras_path=keras_path,
        onnx_path=onnx_path,
    )


def publish_release_to_hf(release: ModelRelease, *, run_id: str) -> str:
    """Publish a staged model release directory to Hugging Face Hub.

    Args:
        release: Staged release object produced by ``stage_model_release``.
        run_id: MLflow run id used in the commit message.

    Returns:
        Hub upload result string, or ``disabled`` when publishing is off.

    Raises:
        RuntimeError: If publishing is enabled without required dependencies or
        credentials.
    """
    if not MLOpsSettings.HF_PUBLISH_ENABLED:
        # Publishing is optional; returning a string keeps the caller's MLflow
        # summary explicit without forcing Hub credentials in every environment.
        return "disabled"
    if not MLOpsSettings.HF_TOKEN:
        raise RuntimeError("HF_PUBLISH_ENABLED requires HF_TOKEN.")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is missing from the MLOps environment.") from exc

    api = HfApi(token=MLOpsSettings.HF_TOKEN)
    # Upload only the model-specific directory. The relative path preserves the
    # saved_models subtree expected by the serving image.
    result = api.upload_folder(
        repo_id=MLOpsSettings.HF_PUBLISH_REPO_ID,
        repo_type=MLOpsSettings.HF_PUBLISH_REPO_TYPE,
        folder_path=str(release.model_dir),
        path_in_repo=release.path_in_repo,
        revision=MLOpsSettings.HF_PUBLISH_REVISION,
        create_pr=MLOpsSettings.HF_PUBLISH_CREATE_PR,
        commit_message=f"Promote retrained model from MLflow run {run_id}",
    )
    return str(result)


def _convert_saved_model(saved_model_dir: Path, onnx_path: Path) -> None:
    """Convert a TensorFlow SavedModel directory to ONNX with tf2onnx.

    Args:
        saved_model_dir: Temporary SavedModel export directory.
        onnx_path: Destination ONNX artifact path.

    Raises:
        RuntimeError: If tf2onnx fails or exceeds the conversion timeout.
    """
    python = Path("/opt/onnx-exporter/bin/python")
    if not python.is_file():
        # Docker images may provide a dedicated exporter interpreter; local runs
        # fall back to the current environment for developer convenience.
        python = Path(sys.executable)
    command = [
        str(python),
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        str(saved_model_dir),
        "--output",
        str(onnx_path),
        "--opset",
        str(MLOpsSettings.ONNX_EXPORT_OPSET),
    ]
    try:
        # Capturing output gives promotion failures actionable conversion logs
        # without streaming noisy TensorFlow messages into Airflow task output.
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"ONNX conversion failed with interpreter '{python}': {details}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ONNX conversion exceeded the 15-minute timeout.") from exc


def _infer_input_shape(model, metadata: dict) -> list[int | None]:
    """Infer the ONNX validation input shape from model or metadata.

    Args:
        model: Keras model being staged for release.
        metadata: Parsed model metadata containing fallback inference shape.

    Returns:
        Four-dimensional input shape with dynamic batch dimension.

    Raises:
        ValueError: If the model has multiple inputs.
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        # The inference service accepts one image tensor; multi-input artifacts
        # would need a different serving and validation contract.
        if len(input_shape) != 1:
            raise ValueError("Only single-input models are supported for ONNX release.")
        input_shape = input_shape[0]
    if len(input_shape) == 4 and all(dim is not None for dim in input_shape[1:]):
        return [None, *[int(dim) for dim in input_shape[1:]]]

    # Some Keras models keep dynamic input shapes. Metadata provides the serving
    # dimensions used by the API and retraining pipelines.
    height, width = metadata.get("inference", {}).get("input_size", [256, 256])
    channels = metadata.get("inference", {}).get("channels", 3)
    return [None, int(height), int(width), int(channels)]


def _validate_onnx(model, onnx_path: Path, input_shape, ort) -> dict[str, object]:
    """Compare Keras and ONNX outputs on a deterministic random sample.

    The deterministic sample is not a clinical evaluation. It is a release
    safety check that catches broken exports, channel-order mistakes, and gross
    numerical mismatches before publication.
    """
    shape = [1 if dim is None else int(dim) for dim in input_shape]
    # A fixed RNG sample makes validation reproducible across CI, Airflow, and
    # local promotion runs.
    sample = np.random.default_rng(42).uniform(0, 255, size=shape).astype(np.float32)
    keras_output = np.asarray(model(sample, training=False))

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_output = np.asarray(session.run(None, {input_name: sample})[0])
    # Both absolute error metrics are recorded so model cards and run summaries
    # can show how close the converted artifact stayed to the source model.
    max_abs = float(np.max(np.abs(keras_output - onnx_output)))
    mean_abs = float(np.mean(np.abs(keras_output - onnx_output)))
    valid = bool(
        np.allclose(
            keras_output,
            onnx_output,
            rtol=MLOpsSettings.ONNX_VALIDATION_RTOL,
            atol=MLOpsSettings.ONNX_VALIDATION_ATOL,
        )
    )
    if not valid:
        # Failing fast here prevents a numerically invalid ONNX artifact from
        # being published to the runtime model repository.
        raise RuntimeError(
            f"ONNX output validation failed: max_abs={max_abs}, mean_abs={mean_abs}"
        )
    return {
        "valid": True,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rtol": MLOpsSettings.ONNX_VALIDATION_RTOL,
        "atol": MLOpsSettings.ONNX_VALIDATION_ATOL,
    }
