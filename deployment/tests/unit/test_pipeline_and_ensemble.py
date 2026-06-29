from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.predictor.classification_onnx import EnsembleBinaryOnnxClassifier
from app.predictor.pipeline import LungDetection


class StubBinaryModel:
    class_map = {0: "Healthy", 1: "Unhealthy"}

    def __init__(self, probability):
        self.probability = probability

    def predict(self, img, mask):
        label = int(self.probability >= 0.5)
        return self.probability, label, self.class_map[label]


def test_ensemble_averages_probabilities_and_returns_model_details():
    ensemble = EnsembleBinaryOnnxClassifier(
        {
            "first": StubBinaryModel(0.2),
            "second": StubBinaryModel(0.8),
        }
    )

    result = ensemble.predict(img=None, mask=None, return_all=True)

    assert result["final_prob"] == pytest.approx(0.5)
    assert result["final_label"] == 1
    assert result["final_label_name"] == "Unhealthy"
    assert set(result["models_results"]) == {"first", "second"}


def _detector_with_binary_result(final_label):
    detector = LungDetection.__new__(LungDetection)
    detector.seg_model = Mock()
    detector.seg_model.predict_mask.return_value = np.ones((4, 4, 1), dtype=np.float32)
    detector.ensemble = Mock()
    detector.ensemble.predict.return_value = {
        "final_prob": float(final_label),
        "final_probs_by_label": {
            "healthy": float(1 - final_label),
            "unhealthy": float(final_label),
        },
        "final_label": final_label,
        "final_label_name": "Unhealthy" if final_label else "Healthy",
    }
    detector.disease_model = Mock()
    detector.disease_model.predict.return_value = {
        "label": 0,
        "label_name": "COVID",
        "probs_by_label": {"COVID": 0.9},
    }
    return detector


@patch("app.predictor.pipeline.crop_lung_roi")
def test_healthy_prediction_does_not_run_disease_model(crop_roi):
    crop_roi.return_value = np.zeros((4, 4, 3), dtype=np.float32)
    detector = _detector_with_binary_result(final_label=0)

    result = detector.predict(np.zeros((1, 4, 4, 3), dtype=np.float32))

    assert "disease" not in result
    detector.disease_model.predict.assert_not_called()


@patch("app.predictor.pipeline.crop_lung_roi")
def test_unhealthy_prediction_runs_disease_model(crop_roi):
    roi = np.zeros((4, 4, 3), dtype=np.float32)
    crop_roi.return_value = roi
    detector = _detector_with_binary_result(final_label=1)

    result = detector.predict(np.zeros((1, 4, 4, 3), dtype=np.float32))

    assert result["disease"]["label_name"] == "COVID"
    detector.disease_model.predict.assert_called_once_with(roi)
