from app.preprocessing.mask import binary_mask_to_rgb_batch
import numpy as np

from app.preprocessing.roi import crop_lung_roi
from app.preprocessing.transforms import (
    apply_mask,
    fill_background,
    normalize_image,
    concat_channels,
    ensure_batch,
)
from typing import List, Callable, Dict
from app.utils.errors import PreprocessError


PREPROCESS_MAP = {
    "inception_v3": lambda img: _preprocess_tf_mode(img),
    "mobilenet_v3": lambda img: np.asarray(img, dtype=np.float32),
    "densenet": lambda img: _preprocess_torch_mode(img),
}


def _preprocess_tf_mode(img):
    return (np.asarray(img, dtype=np.float32) / 127.5) - 1.0


def _preprocess_torch_mode(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std
# -----------------------------
# Pipeline Builder
# -----------------------------
def build_pipeline(config: Dict) -> List[Callable]:
    """
    Build a preprocessing pipeline based on a config dictionary.

    Config example:
    {
        "use_roi": True,
        "roi_target_size": (256, 256),
        "normalize": "imagenet",  # "imagenet", "[-1,1]", "none"
        "use_mask": True,
        "background_fill": -1.0,
        "concat_mask_channels": False
    }
    """
    steps = []

    if config.get("mask_as_rgb", False):
        steps.append(lambda img, mask: (img, binary_mask_to_rgb_batch(mask)))

    if config.get("mask_as_rgb") and config.get("concat_mask_channels"):
        raise PreprocessError(
            "INVALID_PIPELINE_CONFIG",
            "mask_as_rgb and concat_mask_channels cannot both be true.",
        )


    if config.get("use_roi", False):
        target_size = config.get("roi_target_size", (256, 256))
        steps.append(lambda img, mask: crop_lung_roi(img, mask, target_size))

    if config.get("preprocess_input_fn"):
        fn_key = config["preprocess_input_fn"]
        if fn_key not in PREPROCESS_MAP:
            raise PreprocessError(
                "UNKNOWN_PREPROCESS_FN",
                "Unknown preprocess_input_fn.",
                {"requested": fn_key, "available": sorted(PREPROCESS_MAP.keys())},
            )
        fn = PREPROCESS_MAP[fn_key]
        steps.append(lambda img, mask: fn(img))

    if config.get("normalize", None):
        norm_mode = config["normalize"]
        steps.append(lambda img, mask: normalize_image(img, norm_mode))

    if config.get("use_mask", False):
        steps.append(lambda img, mask: apply_mask(img, mask))

    if "background_fill" in config and config["background_fill"] is not None:
        value = config["background_fill"]
        steps.append(lambda img, mask: fill_background(img, mask, value))

    if config.get("concat_mask_channels", False):
        steps.append(lambda img, mask: concat_channels(img, mask))

    return steps


# -----------------------------
# Run Pipeline
# -----------------------------
def run_pipeline(img, mask, steps: List[Callable]) -> np.ndarray:
    """
    Apply a series of preprocessing steps to an image & mask.
    
    Args:
        img: np.ndarray, shape (H, W, C) or (1, H, W, C)
        mask: np.ndarray, shape (H, W) or (H, W, 1)
        steps: List of callables of form func(img, mask) -> img
    
    Returns:
        np.ndarray: preprocessed image
    """
    for step in steps:
        result = step(img, mask)
        if isinstance(result, tuple):
            img, mask = result
        else:
            img = result
    return ensure_batch(img)
