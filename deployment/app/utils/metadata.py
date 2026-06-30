"""Model metadata loading and validation helpers."""

from pathlib import Path

import yaml

from app.configs.config import AppConfig
from app.utils.errors import ArtifactError

def load_metadata(model_dir: Path) -> dict:
    """Load a model directory's YAML metadata file.

    Args:
        model_dir: Directory containing the runtime model artifact and
            ``metadata.yaml``.

    Returns:
        Parsed metadata dictionary.

    Raises:
        ArtifactError: If the metadata file is missing or not valid YAML.
    """
    metadata_path = AppConfig.get_metadata_path(model_dir)

    if not metadata_path.exists():
        raise ArtifactError(
            "METADATA_NOT_FOUND",
            "Metadata file not found.",
            {"path": str(metadata_path)},
        )

    try:
        # Safe loading is enough for the project metadata contract and avoids
        # executing arbitrary YAML constructors from artifact files.
        with metadata_path.open("r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ArtifactError(
            "METADATA_INVALID",
            "Metadata file is not valid YAML.",
            {"path": str(metadata_path)},
        ) from exc
