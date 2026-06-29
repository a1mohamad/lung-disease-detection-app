from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.errors import register_exception_handlers
from app.api.routes import router


def _prediction_response():
    return {
        "final_prob": 0.2,
        "final_probs_by_label": {"healthy": 0.8, "unhealthy": 0.2},
        "final_label": 0,
        "final_label_name": "Healthy",
        "source_url": "/static/predictions/source.png",
        "mask_url": "/static/predictions/mask.png",
    }


def _client():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.state.detector = object()
    return TestClient(app)


def test_health_endpoint():
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "name": None,
        "status": "ok",
        "version": "1.0.0",
    }


@patch("app.api.routes.run_inference")
def test_predict_endpoint_passes_json_input_to_inference(run_inference):
    run_inference.return_value = _prediction_response()

    response = _client().post(
        "/predict?return_all=false",
        json={"image_url": "https://example.com/xray.png"},
    )

    assert response.status_code == 200
    assert response.json()["final_label_name"] == "Healthy"
    run_inference.assert_called_once()
    call = run_inference.call_args.kwargs
    assert call["image_url"] == "https://example.com/xray.png"
    assert call["return_all"] is False


def test_predict_endpoint_rejects_multiple_sources():
    response = _client().post(
        "/predict",
        json={
            "image_path": "scan.png",
            "image_url": "https://example.com/xray.png",
        },
    )

    assert response.status_code == 422


@patch("app.api.routes.run_inference")
def test_upload_endpoint_passes_file_to_inference(run_inference):
    run_inference.return_value = _prediction_response()

    response = _client().post(
        "/predict/upload",
        files={"file": ("xray.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200
    call = run_inference.call_args.kwargs
    assert call["upload_file"] is not None
    assert call["return_all"] is True
