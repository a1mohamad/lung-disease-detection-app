import numpy as np

from app.utils.errors import PreprocessError


def ensure_batch(x):
    """Ensure array has batch dimension."""
    arr = np.asarray(x)
    return arr if arr.ndim == 4 else np.expand_dims(arr, axis=0)


def ensure_channel(x):
    """Ensure array has channel dimension."""
    arr = np.asarray(x)
    return arr if arr.ndim >= 3 else np.expand_dims(arr, axis=-1)


def apply_mask(img, mask):
    """Apply binary mask to image."""
    img = ensure_batch(img)
    mask = ensure_batch(mask)
    return img * mask


def invert_mask(mask):
    """Invert a binary mask."""
    mask = ensure_batch(mask)
    return 1.0 - mask


def fill_background(
    img,
    mask,
    value: float = -1.0
) -> np.ndarray:
    """Fill background area with constant value."""
    img = ensure_batch(img)
    mask = ensure_batch(mask)
    background = invert_mask(mask) * value
    masked_img = apply_mask(img, mask)
    return masked_img + background


def normalize_image(img, mode: str) -> np.ndarray:
    """
    Normalize image array.

    Modes:
        - 'imagenet' : divide by 255
        - '[-1,1]'   : scale to [-1, 1]
        - 'none'     : no normalization
    """
    img = np.asarray(img, dtype=np.float32)

    if mode == "imagenet":
        return img / 255.0
    if mode == "[-1,1]":
        return (img / 127.5) - 1.0
    if mode == "none":
        return img

    raise PreprocessError(
        "UNKNOWN_NORMALIZATION",
        "Unknown normalization mode.",
        {"mode": mode},
    )


def concat_channels(
    img,
    extra,
) -> np.ndarray:
    """Concatenate arrays along channel axis."""
    img = ensure_batch(img)
    extra = ensure_batch(extra)
    extra = ensure_channel(extra)
    return np.concatenate([img, extra], axis=-1)
