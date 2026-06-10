from __future__ import annotations

import numpy as np

from app.configs.constants import IMAGENET_TORCH_MEAN, IMAGENET_TORCH_STD


def preprocess_tf_mode(img) -> np.ndarray:
    return (np.asarray(img, dtype=np.float32) / 127.5) - 1.0


def preprocess_torch_mode(img) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr - IMAGENET_TORCH_MEAN) / IMAGENET_TORCH_STD


def preprocess_identity_float32(img) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


PREPROCESS_MAP = {
    "inception_v3": preprocess_tf_mode,
    "mobilenet_v3": preprocess_identity_float32,
    "densenet": preprocess_torch_mode,
}
