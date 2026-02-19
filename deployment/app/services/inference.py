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
    image_source, raw_bytes = _select_image_source(
        image_path=image_path,
        image_url=image_url,
        image_base64=image_base64,
        upload_file=upload_file,
    )

    img = load_image(raw_bytes, target_size=AppConfig.IMAGE_SIZE)
    results = detector.predict(img, return_all=return_all)
    binary = results["binary"]

    response: Dict[str, Any] = {
        "final_prob": binary["final_prob"],
        "final_probs_by_label": binary["final_probs_by_label"],
        "final_label": binary["final_label"],
        "final_label_name": binary.get("final_label_name"),
    }

    if return_all and "models_results" in binary:
        response["models_results"] = binary["models_results"]

    if "disease" in results:
        response["disease"] = results["disease"]

    source_img = _bytes_to_np(raw_bytes)
    mask = results["mask"].numpy()
    roi = results["roi_img"].numpy()

    response.update(
        save_output_images(
            source_image=source_img,
            mask=mask,
            roi_img=roi,
        )
    )

    return response
