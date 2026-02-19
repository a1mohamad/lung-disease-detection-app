from pathlib import Path

import yaml

from app.configs.config import AppConfig
from app.utils.errors import ArtifactError

def load_metadata(model_dir: Path) -> dict:
    metadata_path = AppConfig.get_metadata_path(model_dir)

    if not metadata_path.exists():
        raise ArtifactError(
            "METADATA_NOT_FOUND",
            "Metadata file not found.",
            {"path": str(metadata_path)},
        )

    try:
        with metadata_path.open("r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ArtifactError(
            "METADATA_INVALID",
            "Metadata file is not valid YAML.",
            {"path": str(metadata_path)},
        ) from exc
