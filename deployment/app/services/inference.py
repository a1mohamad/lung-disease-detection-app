"""Service-level orchestration for one prediction request."""

from __future__ import annotations

from typing import IO, Optional, Any, Dict
from app.configs.config import AppConfig
from app.preprocessing.image import load_image
from app.predictor.pipeline import LungDetection
from app.services.input import _select_image_source, _bytes_to_np
from app.services.outputs import save_output_images


def run_inference(
    *,
    detector: LungDetection,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    upload_file: Optional[IO[bytes]] = None,
    return_all: bool = True,
) -> Dict[str, Any]:
    """Execute one end-to-end inference request for the API layer.

    This service bridges transport-specific inputs and the model pipeline. It
    validates the requested image source, converts bytes into the model tensor,
    runs the detector, saves visual artifacts, and returns the public response
    contract consumed by JSON, upload, logging, and Kafka paths.

    Args:
        detector: Process-wide ``LungDetection`` instance created at startup.
        image_path: Optional local image path.
        image_url: Optional remote image URL.
        image_base64: Optional base64 image payload.
        upload_file: Optional multipart upload stream.
        return_all: Include per-model binary ensemble details when true.

    Returns:
        API-ready prediction dictionary with labels, probabilities, optional
        disease subtype, and generated artifact paths or URLs.
    """
    # Source selection centralizes the mutually exclusive path/url/base64/upload
    # rules so route handlers can stay thin and consistent.
    image_source, raw_bytes = _select_image_source(
        image_path=image_path,
        image_url=image_url,
        image_base64=image_base64,
        upload_file=upload_file,
    )

    img = load_image(raw_bytes, target_size=AppConfig.IMAGE_SIZE)
    # The detector returns internal pipeline outputs; this service translates
    # them into the stable public response contract.
    results = detector.predict(img, return_all=return_all)
    binary = results["binary"]

    response: Dict[str, Any] = {
        "final_prob": binary["final_prob"],
        "final_probs_by_label": binary["final_probs_by_label"],
        "final_label": binary["final_label"],
        "final_label_name": binary.get("final_label_name"),
    }

    if return_all and "models_results" in binary:
        # Per-model outputs are optional to keep lightweight clients from
        # receiving ensemble internals unless requested.
        response["models_results"] = binary["models_results"]

    if "disease" in results:
        # Disease subtype results appear only when the binary ensemble selected
        # the unhealthy branch.
        response["disease"] = results["disease"]

    source_img = _bytes_to_np(raw_bytes)
    mask = results["mask"]
    roi = results["roi_img"]

    response.update(
        # Artifact saving happens after prediction so masks and ROI crops can be
        # linked in the same response and persisted by the logging layer.
        save_output_images(
            source_image=source_img,
            mask=mask,
            roi_img=roi,
        )
    )

    return response
