"""Mask format conversion helpers used by preprocessing pipelines."""

import numpy as np
from app.preprocessing.transforms import ensure_batch, ensure_channel


def binary_mask_to_rgb_batch(binary_mask) -> np.ndarray:
    """Convert a binary mask to batched three-channel RGB format.

    Args:
        binary_mask: Mask array in 2D, single-channel, three-channel, or batched
            format.

    Returns:
        Mask array with shape ``(1, H, W, 3)``.

    Notes:
        Some classifier experiments used mask-as-RGB inputs. This helper keeps
        that training-time contract available in metadata-driven preprocessing.
    """
    # Classifier branches that were trained on RGB masks expect three channels.
    mask = ensure_channel(binary_mask)  # (H, W, 1) or (1, H, W, 1)

    if mask.ndim == 4 and mask.shape[-1] == 1:
        mask_rgb = np.repeat(mask, 3, axis=-1)
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask_rgb = np.repeat(mask, 3, axis=-1)
    elif mask.ndim >= 3 and mask.shape[-1] == 3:
        mask_rgb = mask
    else:
        mask_rgb = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)

    # The downstream classifier wrappers expect a batch dimension regardless of
    # whether preprocessing received a single mask or an already batched mask.
    mask_rgb = ensure_batch(mask_rgb)   # (1, H, W, 3)

    return mask_rgb
