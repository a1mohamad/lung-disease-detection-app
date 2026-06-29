import numpy as np
import pytest
from pydantic import ValidationError

from app.preprocessing.pipeline import build_pipeline, run_pipeline
from app.preprocessing.roi import crop_lung_roi
from app.preprocessing.transforms import normalize_image
from app.schemas.request import PredictRequest
from app.utils.errors import PreprocessError


def test_predict_request_accepts_exactly_one_source():
    request = PredictRequest(image_url="https://example.com/xray.png")

    assert request.image_url == "https://example.com/xray.png"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"image_path": "scan.png", "image_url": "https://example.com/scan.png"},
    ],
)
def test_predict_request_rejects_missing_or_multiple_sources(payload):
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("imagenet", 1.0),
        ("[-1,1]", 1.0),
        ("none", 255.0),
    ],
)
def test_normalize_image_modes(mode, expected):
    image = np.full((1, 2, 2, 3), 255, dtype=np.float32)

    result = normalize_image(image, mode)

    assert result.dtype == np.float32
    assert np.allclose(result, expected)


def test_unknown_normalization_is_rejected():
    with pytest.raises(PreprocessError) as exc_info:
        normalize_image(np.zeros((2, 2, 3)), "unknown")

    assert exc_info.value.error_code == "UNKNOWN_NORMALIZATION"


def test_pipeline_rejects_conflicting_mask_configuration():
    with pytest.raises(PreprocessError) as exc_info:
        build_pipeline({"mask_as_rgb": True, "concat_mask_channels": True})

    assert exc_info.value.error_code == "INVALID_PIPELINE_CONFIG"


def test_pipeline_can_append_mask_as_fourth_channel():
    image = np.ones((2, 2, 3), dtype=np.float32)
    mask = np.ones((2, 2, 1), dtype=np.float32)
    steps = build_pipeline({"concat_mask_channels": True})

    result = run_pipeline(image, mask, steps)

    assert result.shape == (1, 2, 2, 4)


def test_empty_mask_roi_falls_back_to_full_image():
    image = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    mask = np.zeros((4, 6, 1), dtype=np.float32)

    result = crop_lung_roi(image, mask, target_size=(4, 6))

    assert result.shape == (4, 6, 3)
    assert np.allclose(result, image)
