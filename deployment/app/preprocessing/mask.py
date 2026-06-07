import numpy as np
from app.preprocessing.transforms import ensure_batch, ensure_channel


def binary_mask_to_rgb_batch(binary_mask) -> np.ndarray:
    """
    Converts a binary mask to 3-channel format and adds batch dimension.
    Output shape: (1, H, W, 3)
    """
    # Ensure mask has channel dimension
    mask = ensure_channel(binary_mask)  # (H, W, 1) or (1, H, W, 1)

    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask_rgb = np.repeat(mask, 3, axis=-1)
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask_rgb = np.repeat(mask, 3, axis=-1)
    elif mask.ndim >= 3 and mask.shape[-1] == 3:
        mask_rgb = mask
    else:
        mask_rgb = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

    # Ensure batch dimension
    mask_rgb = ensure_batch(mask_rgb)   # (1, H, W, 3)

    return mask_rgb
