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
    root: Path
    model_dir: Path
    path_in_repo: str
    keras_path: Path
    onnx_path: Path


def stage_model_release(*, model, spec, metadata: dict, run_id: str) -> ModelRelease:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX release dependencies are missing from the MLOps environment."
        ) from exc

    relative_model_dir = spec.model_dir.relative_to(SAVED_MODELS_DIR)
    release_root = MLOpsSettings.MODEL_RELEASE_ROOT / run_id
    model_dir = release_root / relative_model_dir
    if release_root.exists():
        raise RuntimeError(f"Model release directory already exists: {release_root}")
    model_dir.mkdir(parents=True)

    model_config = metadata.get("model", {})
    keras_name = model_config.get("path")
    onnx_name = model_config.get("onnx_path")
    if not keras_name or not onnx_name:
        raise ValueError(f"Model metadata lacks Keras or ONNX path: {spec.metadata_path}")

    keras_path = model_dir / keras_name
    onnx_path = model_dir / onnx_name
    model.save(str(keras_path))
    shutil.copy2(spec.metadata_path, model_dir / "metadata.yaml")

    input_shape = _infer_input_shape(model, metadata)
    saved_model_dir = release_root / ".onnx_saved_model"
    model.export(str(saved_model_dir))
    _convert_saved_model(saved_model_dir, onnx_path)
    validation = _validate_onnx(model, onnx_path, input_shape, ort)
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
    if not MLOpsSettings.HF_PUBLISH_ENABLED:
        return "disabled"
    if not MLOpsSettings.HF_TOKEN:
        raise RuntimeError("HF_PUBLISH_ENABLED requires HF_TOKEN.")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is missing from the MLOps environment.") from exc

    api = HfApi(token=MLOpsSettings.HF_TOKEN)
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
    python = Path("/opt/onnx-exporter/bin/python")
    if not python.is_file():
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
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError("Only single-input models are supported for ONNX release.")
        input_shape = input_shape[0]
    if len(input_shape) == 4 and all(dim is not None for dim in input_shape[1:]):
        return [None, *[int(dim) for dim in input_shape[1:]]]

    height, width = metadata.get("inference", {}).get("input_size", [256, 256])
    channels = metadata.get("inference", {}).get("channels", 3)
    return [None, int(height), int(width), int(channels)]


def _validate_onnx(model, onnx_path: Path, input_shape, ort) -> dict[str, object]:
    shape = [1 if dim is None else int(dim) for dim in input_shape]
    sample = np.random.default_rng(42).uniform(0, 255, size=shape).astype(np.float32)
    keras_output = np.asarray(model(sample, training=False))

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    onnx_output = np.asarray(session.run(None, {input_name: sample})[0])
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
