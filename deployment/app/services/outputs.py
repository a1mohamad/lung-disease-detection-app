from app.configs.config import AppConfig
from app.utils.visualization import overlay_mask_on_image

import numpy as np
from pathlib import Path
from PIL import Image
from uuid import uuid4

def save_output_images(
    *,
    source_image: np.ndarray,
    mask: np.ndarray,
    roi_img: np.ndarray,
    prefix: str | None = None,
) -> dict:
    """
    Save source, mask, roi, and overlay images.
    Returns dict with URLs/paths.
    """
    pred_dir = AppConfig.PREDICTION_DIR
    pred_dir.mkdir(parents=True, exist_ok=True)
    uid = prefix or uuid4().hex

    source_path = pred_dir / f"{uid}_source.png"
    mask_path = pred_dir / f"{uid}_mask.png"
    roi_path = pred_dir / f"{uid}_roi.png"
    overlay_path = pred_dir / f"{uid}_overlay.png"

    _save_source_image(source_image, source_path)
    _save_mask(mask, mask_path)
    _save_roi(roi_img, roi_path)
    _save_overlay_image_mask(source_image, mask, overlay_path)

    return {
        "source_url": f"/static/predictions/{source_path.name}",
        "mask_url": f"/static/predictions/{mask_path.name}",
        "roi_url": f"/static/predictions/{roi_path.name}",
        "overlay_url": f"/static/predictions/{overlay_path.name}",
    }

def _save_mask(mask: np.ndarray, path: Path) -> None:
    mask_2d = np.squeeze(mask)
    if mask_2d.dtype != np.uint8:
        mask_2d = (mask_2d > 0).astype("uint8") * 255
    Image.fromarray(mask_2d, mode="L").save(path)

def _save_roi(roi_img: np.ndarray, path: Path) -> None:
    roi_img = np.squeeze(roi_img)
    if roi_img.dtype != np.uint8:
        roi_img = roi_img.astype("uint8")
    Image.fromarray(roi_img).save(path)

def _save_source_image(img: np.ndarray, path: Path) -> None:
    arr = img.astype("uint8")
    Image.fromarray(arr).save(path)

def _save_overlay_image_mask(
    img: np.ndarray, 
    mask: np.ndarray,
    path: Path) -> None:

    overlay = overlay_mask_on_image(img, mask)
    overlay.save(path)

