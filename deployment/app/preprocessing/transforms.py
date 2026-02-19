import tensorflow as tf

from app.utils.errors import PreprocessError


def ensure_batch(x: tf.Tensor) -> tf.Tensor:
    """Ensure tensor has batch dimension."""
    return x if x.ndim == 4 else tf.expand_dims(x, axis=0)


def ensure_channel(x: tf.Tensor) -> tf.Tensor:
    """Ensure tensor has channel dimension."""
    return x if x.ndim >= 3 else tf.expand_dims(x, axis=-1)


def apply_mask(img: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Apply binary mask to image."""
    img = ensure_batch(img)
    mask = ensure_batch(mask)
    return img * mask


def invert_mask(mask: tf.Tensor) -> tf.Tensor:
    """Invert a binary mask."""
    mask = ensure_batch(mask)
    return 1.0 - mask


def fill_background(
    img: tf.Tensor,
    mask: tf.Tensor,
    value: float = -1.0
) -> tf.Tensor:
    """Fill background area with constant value."""
    img = ensure_batch(img)
    mask = ensure_batch(mask)
    background = invert_mask(mask) * value
    masked_img = apply_mask(img, mask)
    return masked_img + background


def normalize_image(img: tf.Tensor, mode: str) -> tf.Tensor:
    """
    Normalize image tensor.

    Modes:
        - 'imagenet' : divide by 255
        - '[-1,1]'   : scale to [-1, 1]
        - 'none'     : no normalization
    """
    img = tf.cast(img, tf.float32)

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
    img: tf.Tensor,
    extra: tf.Tensor
) -> tf.Tensor:
    """Concatenate tensors along channel axis."""
    img = ensure_batch(img)
    extra = ensure_batch(extra)
    extra = ensure_channel(extra)
    return tf.concat([img, extra], axis=-1)
