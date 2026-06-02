from app.configs.config import AppConfig
from app.utils.visualization import overlay_mask_on_image

import numpy as np
from datetime import datetime, timezone
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

    Files are grouped one folder per prediction, under a UTC day folder, e.g.
    ``predictions/2026-06-02/<uid>/source.png`` so the output stays ordered
    instead of a flat dump. Returns dict with the matching static URLs.
    """
    uid = prefix or uuid4().hex
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rel_dir = f"{day}/{uid}"
    out_dir = AppConfig.PREDICTION_DIR / day / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = out_dir / "source.png"
    mask_path = out_dir / "mask.png"
    roi_path = out_dir / "roi.png"
    overlay_path = out_dir / "overlay.png"

    _save_source_image(source_image, source_path)
    _save_mask(mask, mask_path)
    _save_roi(roi_img, roi_path)
    _save_overlay_image_mask(source_image, mask, overlay_path)

    return {
        "source_url": f"/static/predictions/{rel_dir}/{source_path.name}",
        "mask_url": f"/static/predictions/{rel_dir}/{mask_path.name}",
        "roi_url": f"/static/predictions/{rel_dir}/{roi_path.name}",
        "overlay_url": f"/static/predictions/{rel_dir}/{overlay_path.name}",
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

