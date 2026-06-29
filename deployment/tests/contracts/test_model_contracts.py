import json

import pytest
import yaml

from app.configs.config import AppConfig
from app.utils.errors import ArtifactError
from app.utils.metadata import load_metadata
from app.utils.onnx_loader import get_onnx_model_path


def test_checked_in_class_mappings_match_api_labels():
    classification = json.loads(AppConfig.CLASSIFICATION_JSON.read_text())
    diseases = json.loads(AppConfig.DISEASES_JSON.read_text())

    assert classification == {"0": "Healthy", "1": "Unhealthy"}
    assert diseases == {
        "0": "COVID",
        "1": "Viral Pneumonia",
        "2": "Lung Opacity",
    }


def test_load_metadata_reads_valid_yaml(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "metadata.yaml").write_text(
        yaml.safe_dump({"model": {"path": "model.keras"}}),
        encoding="utf-8",
    )

    metadata = load_metadata(model_dir)

    assert metadata["model"]["path"] == "model.keras"


def test_load_metadata_reports_missing_file(tmp_path):
    with pytest.raises(ArtifactError) as exc_info:
        load_metadata(tmp_path)

    assert exc_info.value.error_code == "METADATA_NOT_FOUND"


def test_onnx_path_prefers_explicit_name_and_has_keras_fallback(tmp_path):
    explicit = get_onnx_model_path(
        tmp_path,
        {"model": {"path": "model.keras", "onnx_path": "export/model.onnx"}},
    )
    fallback = get_onnx_model_path(
        tmp_path,
        {"model": {"path": "model.keras"}},
    )

    assert explicit == tmp_path / "export" / "model.onnx"
    assert fallback == tmp_path / "model.onnx"
