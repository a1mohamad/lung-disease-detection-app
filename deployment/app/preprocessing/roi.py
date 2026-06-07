import numpy as np
from PIL import Image

from app.preprocessing.mask import binary_mask_to_rgb_batch
from app.preprocessing.transforms import ensure_batch

def crop_lung_roi(
    img,
    mask,
    target_size: tuple,
    threshold: float = 0.5,
    margin_ratio: float = 0.1,
) -> np.ndarray:
    """
    Crop lung region using a binary segmentation mask and resize.

    Args:
        img: np.ndarray, shape (1, H, W, 3) or (H, W, 3)
        mask: np.ndarray, shape (H, W), (H, W, 1), or (1, H, W, 1)
        target_size: tuple (H, W)
        threshold: float, binarization threshold for mask
        margin_ratio: float, extra margin around lung bbox

    Returns:
        np.ndarray: Cropped and resized image, shape (target_H, target_W, 3)
    """

    # --- Ensure proper shapes using transforms ---
    img = ensure_batch(img).astype(np.float32)       # shape: (1, H, W, C)
    mask_rgb = binary_mask_to_rgb_batch(mask)  # (1, H, W, 3)

    # --- Binarize mask ---
    mask_2d = mask_rgb[0, :, :, 0] > threshold  # drop batch for indexing
    indices = np.argwhere(mask_2d)

    if indices.size > 0:
        y_min, x_min = indices.min(axis=0)
        y_max, x_max = indices.max(axis=0)

        h = int(y_max - y_min)
        w = int(x_max - x_min)

        margin_y = int(max(h * margin_ratio, 1.0))
        margin_x = int(max(w * margin_ratio, 1.0))

        img_h, img_w = img.shape[1:3]
        y_start = max(0, int(y_min) - margin_y)
        x_start = max(0, int(x_min) - margin_x)
        y_end = min(img_h, int(y_max) + margin_y)
        x_end = min(img_w, int(x_max) + margin_x)
        cropped = img[0, y_start:y_end, x_start:x_end, :]
    else:
        cropped = img[0]

    return _resize_float_image(cropped, target_size)


def _resize_float_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    pil_img = Image.fromarray(np.clip(img, 0, 255).astype("uint8"))
    # PIL expects (width, height), while model metadata uses (height, width).
    resized = pil_img.resize((int(target_size[1]), int(target_size[0])), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)
