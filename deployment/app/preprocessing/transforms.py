"""Small NumPy preprocessing transforms used by metadata pipelines."""

import numpy as np

from app.utils.errors import PreprocessError


def ensure_batch(x):
    """Ensure an array has a leading batch dimension.

    Args:
        x: Image or mask-like array.

    Returns:
        Array with shape ``(N, ...)``. Four-dimensional arrays are returned
        unchanged because they are already batched.
    """
    arr = np.asarray(x)
    # Classifier and segmentation models accept batched tensors; a single image
    # is promoted to a batch of one while existing batches pass through.
    return arr if arr.ndim == 4 else np.expand_dims(arr, axis=0)


def ensure_channel(x):
    """Ensure an array has an explicit channel dimension.

    Args:
        x: Mask or image-like array.

    Returns:
        Array with a final channel axis when the input is two-dimensional.
    """
    arr = np.asarray(x)
    # Masks commonly arrive as HxW arrays. Channel-last expansion keeps them
    # broadcast-compatible with RGB image batches.
    return arr if arr.ndim >= 3 else np.expand_dims(arr, axis=-1)


def apply_mask(img, mask):
    """Apply a binary mask to a batched image.

    Args:
        img: Image array, batched or unbatched.
        mask: Binary mask array, batched or unbatched.

    Returns:
        Image with background pixels removed by multiplication.
    """
    img = ensure_batch(img)
    mask = ensure_batch(mask)
    # NumPy broadcasting handles single-channel masks over RGB image channels.
    return img * mask


def invert_mask(mask):
    """Invert a binary mask.

    Args:
        mask: Binary mask array where foreground is represented by ``1``.

    Returns:
        Batched inverse mask where foreground and background are swapped.
    """
    mask = ensure_batch(mask)
    # Masks are binary floats in this pipeline, so inversion is a simple
    # foreground/background swap.
    return 1.0 - mask


def fill_background(
    img,
    mask,
    value: float = -1.0
) -> np.ndarray:
    """Fill masked-out image regions with a constant value.

    Args:
        img: Image array, batched or unbatched.
        mask: Binary foreground mask.
        value: Fill value used for background pixels.

    Returns:
        Image where lung/foreground pixels are preserved and background pixels
        are replaced by ``value``.
    """
    img = ensure_batch(img)
    mask = ensure_batch(mask)
    # Background fill preserves original tensor shape while replacing pixels
    # outside the predicted lung region.
    background = invert_mask(mask) * value
    masked_img = apply_mask(img, mask)
    return masked_img + background


def normalize_image(img, mode: str) -> np.ndarray:
    """Normalize image values according to a metadata-selected mode.

    Args:
        img: Image array with raw or already-scaled pixel values.
        mode: Normalization mode. Supported values are ``imagenet`` for
            ``[0, 1]`` scaling, ``[-1,1]`` for TensorFlow/Inception scaling,
            and ``none`` to preserve values.

    Returns:
        Float32 image array normalized according to ``mode``.

    Raises:
        PreprocessError: If ``mode`` is not supported.
    """
    img = np.asarray(img, dtype=np.float32)

    if mode == "imagenet":
        # ImageNet-style scaling keeps values in [0, 1] before optional
        # channel-wise preprocessing.
        return img / 255.0
    if mode == "[-1,1]":
        # TensorFlow application models commonly expect values centered at zero.
        return (img / 127.5) - 1.0
    if mode == "none":
        # Explicit no-op mode is useful when metadata wants dtype conversion but
        # the model already handles scaling internally.
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
    """Concatenate an extra channel tensor onto an image batch.

    Args:
        img: Image array.
        extra: Additional channel data, usually a mask.

    Returns:
        Batched array with ``extra`` appended on the final channel axis.
    """
    img = ensure_batch(img)
    extra = ensure_batch(extra)
    extra = ensure_channel(extra)
    # Channel concatenation supports models trained with image+mask inputs.
    return np.concatenate([img, extra], axis=-1)
