"""Region-of-interest cropping based on predicted lung masks."""

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
    """Crop the lung region from an image using a predicted segmentation mask.

    Args:
        img: Image array with shape ``(1, H, W, 3)`` or ``(H, W, 3)``.
        mask: Mask array with shape ``(H, W)``, ``(H, W, 1)``, or
            ``(1, H, W, 1)``.
        target_size: Output size in ``(height, width)`` order.
        threshold: Probability threshold used to binarize the mask.
        margin_ratio: Fractional margin added around the detected lung box.

    Returns:
        Cropped and resized float32 ROI image with shape
        ``(target_height, target_width, 3)``.

    Notes:
        If the mask is empty, the full image is resized. This keeps downstream
        classifiers operational while making the segmentation failure visible
        through the saved mask artifact.
    """

    img = ensure_batch(img).astype(np.float32)       # shape: (1, H, W, C)
    mask_rgb = binary_mask_to_rgb_batch(mask)  # (1, H, W, 3)

    # The bounding box is computed on the first mask channel after batching so
    # callers can pass either raw 2D masks or RGB-expanded masks.
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
        # Empty masks can happen with low-confidence segmentation. Falling back
        # to the full image avoids hiding the event behind a preprocessing crash.
        cropped = img[0]

    return _resize_float_image(cropped, target_size)


def _resize_float_image(img: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resize a float image through PIL while preserving float32 output.

    Args:
        img: Float image array in ``[0, 255]`` scale.
        target_size: Output size in ``(height, width)`` order.

    Returns:
        Resized float32 RGB image.
    """
    pil_img = Image.fromarray(np.clip(img, 0, 255).astype("uint8"))
    # PIL expects (width, height), while model metadata uses (height, width).
    resized = pil_img.resize((int(target_size[1]), int(target_size[0])), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)
