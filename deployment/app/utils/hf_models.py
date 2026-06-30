"""Hugging Face Hub model artifact download helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from app.configs.config import AppConfig
from app.utils.errors import ArtifactError
from app.utils.metadata import load_metadata

logger = logging.getLogger(__name__)


REQUIRED_MODEL_DIRS = (
    AppConfig.UNET_PATH,
    AppConfig.DENSENET_PATH,
    AppConfig.EFFICIENTNET_PATH,
    AppConfig.INCEPTION_PATH,
    AppConfig.MOBILENET_PATH,
    AppConfig.DISEASE_DENSENET_PATH,
)


def ensure_models_available_from_huggingface() -> None:
    """Download model artifacts only when the feature flag is enabled.

    This startup hook lets cloud deployments ship a small image and fetch model
    artifacts at boot, while local development can keep using mounted
    ``saved_models`` directories.
    """
    if not AppConfig.HF_MODEL_DOWNLOAD_ENABLED:
        # Local and CI environments generally mount artifacts directly, so the
        # Hub client is imported only when download mode is explicitly enabled.
        return
    download_models_from_huggingface(skip_if_ready=True)


def download_models_from_huggingface(*, skip_if_ready: bool = True) -> None:
    """Download required model artifacts from the configured Hugging Face repo.

    The download is limited to the model subtrees needed by the inference API.
    After the snapshot completes, local artifacts are revalidated against their
    metadata so a partial or misconfigured Hub repository fails loudly.

    Args:
        skip_if_ready: When true, skip the network download if all local model
            directories already contain valid metadata and model files.

    Raises:
        ArtifactError: If the Hub client is unavailable or the downloaded
        snapshot does not contain the required artifacts.
    """
    if _local_models_ready():
        # Readiness checks validate metadata and runtime artifact paths, not just
        # the existence of top-level model directories.
        if not skip_if_ready:
            logger.info("Local model artifacts already exist; refreshing from Hugging Face.")
        else:
            logger.info("Local model artifacts already exist; skipping Hugging Face download.")
            return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ArtifactError(
            "HF_HUB_NOT_INSTALLED",
            "HF_MODEL_DOWNLOAD_ENABLED=true but huggingface_hub is not installed.",
        ) from exc

    # Scope the snapshot to deployable model folders so large research files do
    # not inflate startup time or container disk usage.
    allow_patterns = _allow_patterns()
    logger.info(
        "Downloading model artifacts from Hugging Face. repo_id=%s revision=%s subdir=%s",
        AppConfig.HF_MODEL_REPO_ID,
        AppConfig.HF_MODEL_REVISION,
        AppConfig.HF_MODEL_REPO_SUBDIR or ".",
    )
    local_dir = _download_target_dir()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=AppConfig.HF_MODEL_REPO_ID,
        repo_type=AppConfig.HF_MODEL_REPO_TYPE,
        revision=AppConfig.HF_MODEL_REVISION,
        token=AppConfig.HF_TOKEN or None,
        allow_patterns=allow_patterns,
        local_dir=local_dir,
    )

    if not _local_models_ready():
        raise ArtifactError(
            "HF_MODEL_DOWNLOAD_INCOMPLETE",
            "Hugging Face model download completed, but required model directories are still missing.",
            {
                "repo_id": AppConfig.HF_MODEL_REPO_ID,
                "repo_subdir": AppConfig.HF_MODEL_REPO_SUBDIR or ".",
                "models_root": str(AppConfig.MODELS_ROOT),
            },
        )


def _local_models_ready() -> bool:
    """Return whether all required model directories contain expected artifacts.

    Returns:
        True when every required model directory has readable metadata and the
        runtime-specific artifact exists.
    """
    for model_dir in REQUIRED_MODEL_DIRS:
        # Metadata is the source of truth for artifact names because Keras and
        # ONNX runtimes use different file extensions and fallback behavior.
        metadata_path = model_dir / "metadata.yaml"
        if not metadata_path.exists():
            return False
        try:
            metadata = load_metadata(model_dir)
        except ArtifactError:
            return False
        model_rel_path = _required_model_rel_path(metadata)
        if not model_rel_path or not (model_dir / model_rel_path).exists():
            return False
    return True


def _required_model_rel_path(metadata: dict) -> str | None:
    """Return the runtime-specific model artifact path from metadata.

    Args:
        metadata: Parsed model metadata.

    Returns:
        Relative artifact path for the configured runtime, or ``None`` when the
        metadata cannot identify one.
    """
    model_cfg = metadata.get("model", {})
    if AppConfig.MODEL_RUNTIME == "onnx":
        # ONNX deployments prefer explicit metadata but can infer the converted
        # filename from a Keras artifact when older metadata lacks onnx_path.
        onnx_rel_path = model_cfg.get("onnx_path")
        if onnx_rel_path:
            return onnx_rel_path
        keras_rel_path = model_cfg.get("path")
        if keras_rel_path:
            return f"{Path(keras_rel_path).stem}.onnx"
        return None
    return model_cfg.get("path")


def _allow_patterns() -> list[str]:
    """Restrict Hub downloads to model artifact subtrees.

    Returns:
        Hugging Face ``allow_patterns`` list scoped to deployable model folders.
    """
    roots = ["healthy_unhealthy/**", "diseases/**", "segmentation/**"]
    if not AppConfig.HF_MODEL_REPO_SUBDIR:
        return roots
    return [f"{AppConfig.HF_MODEL_REPO_SUBDIR}/{pattern}" for pattern in roots]


def _download_target_dir() -> Path:
    """Return the local directory that should receive the Hub snapshot.

    Returns:
        Directory path passed to ``snapshot_download``.
    """
    if AppConfig.HF_MODEL_REPO_SUBDIR:
        return AppConfig.MODELS_ROOT.parent
    return AppConfig.MODELS_ROOT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_models_from_huggingface(skip_if_ready=False)
