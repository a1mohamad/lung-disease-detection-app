"""Model-specific preprocessing functions shared by runtime and MLOps code."""

from __future__ import annotations

import numpy as np

from app.configs.constants import IMAGENET_TORCH_MEAN, IMAGENET_TORCH_STD


def preprocess_tf_mode(img) -> np.ndarray:
    """Scale image values to the TensorFlow/Inception ``[-1, 1]`` range.

    Args:
        img: Image array with raw ``[0, 255]`` values.

    Returns:
        Float32-compatible array scaled to ``[-1, 1]``.
    """
    # Inception-style preprocessing is symmetric around zero and assumes raw
    # image values in the standard 8-bit range.
    return (np.asarray(img, dtype=np.float32) / 127.5) - 1.0


def preprocess_torch_mode(img) -> np.ndarray:
    """Apply ImageNet mean and standard deviation normalization.

    Args:
        img: Image array with raw ``[0, 255]`` values.

    Returns:
        ImageNet-normalized float array.
    """
    # The mean/std arrays are NumPy vectors, so broadcasting applies the same
    # channel-wise normalization to both single images and batches.
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr - IMAGENET_TORCH_MEAN) / IMAGENET_TORCH_STD


def preprocess_identity_float32(img) -> np.ndarray:
    """Return image values as float32 without value scaling.

    Args:
        img: Image array in any NumPy-compatible dtype.

    Returns:
        Float32 image array with original value scale preserved.
    """
    # Some Keras application wrappers perform preprocessing internally; this
    # function keeps those pipelines explicit without changing the tensor scale.
    return np.asarray(img, dtype=np.float32)


# Metadata stores preprocessing by key so model artifacts can declare the exact
# transform they need without importing framework-specific functions directly.
PREPROCESS_MAP = {
    "inception_v3": preprocess_tf_mode,
    "mobilenet_v3": preprocess_identity_float32,
    "densenet": preprocess_torch_mode,
}
