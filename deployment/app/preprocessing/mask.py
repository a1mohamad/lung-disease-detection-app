import tensorflow as tf
from app.preprocessing.transforms import ensure_batch, ensure_channel

def binary_mask_to_rgb_batch(binary_mask: tf.Tensor) -> tf.Tensor:
    """
    Converts a binary mask to 3-channel format and adds batch dimension.
    Output shape: (1, H, W, 3)
    """
    # Ensure mask has channel dimension
    mask = ensure_channel(binary_mask)  # (H, W, 1) or (1, H, W, 1)

    # Convert grayscale to RGB
    mask_rgb = tf.image.grayscale_to_rgb(mask)

    # Ensure batch dimension
    mask_rgb = ensure_batch(mask_rgb)   # (1, H, W, 3)

    return mask_rgb
