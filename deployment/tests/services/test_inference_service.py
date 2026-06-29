from unittest.mock import Mock, patch

import numpy as np

from app.services.inference import run_inference


@patch("app.services.inference.save_output_images")
@patch("app.services.inference._bytes_to_np")
@patch("app.services.inference.load_image")
@patch("app.services.inference._select_image_source")
def test_run_inference_builds_public_response(
    select_source,
    load_image,
    bytes_to_np,
    save_output_images,
):
    raw_bytes = b"image-bytes"
    model_input = np.zeros((1, 4, 4, 3), dtype=np.float32)
    select_source.return_value = ("upload", raw_bytes)
    load_image.return_value = model_input
    bytes_to_np.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    save_output_images.return_value = {
        "source_url": "/static/predictions/source.png",
        "mask_url": "/static/predictions/mask.png",
    }

    detector = Mock()
    detector.predict.return_value = {
        "mask": np.ones((4, 4, 1), dtype=np.float32),
        "roi_img": np.zeros((4, 4, 3), dtype=np.float32),
        "binary": {
            "final_prob": 0.8,
            "final_probs_by_label": {"healthy": 0.2, "unhealthy": 0.8},
            "final_label": 1,
            "final_label_name": "Unhealthy",
            "models_results": {"densenet": {"prob": 0.8}},
        },
        "disease": {
            "label": 0,
            "label_name": "COVID",
            "probs_by_label": {"COVID": 0.9},
        },
    }

    response = run_inference(
        detector=detector,
        upload_file=Mock(),
        return_all=True,
    )

    assert response["final_label_name"] == "Unhealthy"
    assert response["models_results"]["densenet"]["prob"] == 0.8
    assert response["disease"]["label_name"] == "COVID"
    assert response["mask_url"].endswith("mask.png")
    detector.predict.assert_called_once_with(model_input, return_all=True)


@patch("app.services.inference.save_output_images")
@patch("app.services.inference._bytes_to_np")
@patch("app.services.inference.load_image")
@patch("app.services.inference._select_image_source")
def test_run_inference_omits_model_details_when_not_requested(
    select_source,
    load_image,
    bytes_to_np,
    save_output_images,
):
    select_source.return_value = ("upload", b"image-bytes")
    load_image.return_value = np.zeros((1, 2, 2, 3), dtype=np.float32)
    bytes_to_np.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    save_output_images.return_value = {}

    detector = Mock()
    detector.predict.return_value = {
        "mask": np.ones((2, 2, 1), dtype=np.float32),
        "roi_img": np.zeros((2, 2, 3), dtype=np.float32),
        "binary": {
            "final_prob": 0.1,
            "final_probs_by_label": {"healthy": 0.9, "unhealthy": 0.1},
            "final_label": 0,
            "final_label_name": "Healthy",
            "models_results": {"densenet": {"prob": 0.1}},
        },
    }

    response = run_inference(
        detector=detector,
        upload_file=Mock(),
        return_all=False,
    )

    assert "models_results" not in response
    assert "disease" not in response
