"""Metadata-driven preprocessing pipeline builder for model inputs."""

from app.preprocessing.mask import binary_mask_to_rgb_batch
import numpy as np
from app.preprocessing.model_preprocessing import PREPROCESS_MAP
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


def build_pipeline(config: Dict) -> List[Callable]:
    """Build an ordered preprocessing pipeline from model metadata.

    The metadata file attached to each model declares how that artifact expects
    input tensors to be shaped and normalized. This builder converts those
    declarative settings into a list of callables that ``run_pipeline`` can
    execute consistently at inference time.

    Supported configuration keys include ROI cropping, Keras preprocessing
    functions, normalization modes, mask application, background filling, mask
    channel concatenation, and mask-to-RGB conversion.

    Args:
        config: Preprocessing section loaded from ``metadata.yaml``.

    Returns:
        Ordered list of callables with signature ``(img, mask) -> img`` or
        ``(img, mask) -> (img, mask)``.

    Raises:
        PreprocessError: If mutually exclusive options are enabled or an
        unknown preprocessing function is requested.
    """
    steps = []

    if config.get("mask_as_rgb", False):
        # Some models were trained with a three-channel mask image instead of a
        # single binary mask; convert it before downstream image operations.
        steps.append(lambda img, mask: (img, binary_mask_to_rgb_batch(mask)))
    if config.get("mask_as_rgb") and config.get("concat_mask_channels"):
        raise PreprocessError(
            "INVALID_PIPELINE_CONFIG",
            "mask_as_rgb and concat_mask_channels cannot both be true.",
        )


    if config.get("use_roi", False):
        # ROI cropping uses the segmentation mask to focus classifiers on lung
        # tissue while preserving the artifact-specific target size.
        target_size = config.get("roi_target_size", (256, 256))
        steps.append(lambda img, mask: crop_lung_roi(img, mask, target_size))

    if config.get("preprocess_input_fn"):
        # Preprocessing names are validated up front so configuration mistakes
        # fail during pipeline construction, not halfway through inference.
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
        # Generic normalization remains separate from model-specific functions
        # because older artifacts use simple scaling instead of app preprocessors.
        norm_mode = config["normalize"]
        steps.append(lambda img, mask: normalize_image(img, norm_mode))

    if config.get("use_mask", False):
        # Mask application removes background pixels for classifiers trained on
        # lung-only inputs.
        steps.append(lambda img, mask: apply_mask(img, mask))

    if "background_fill" in config and config["background_fill"] is not None:
        # Filling background with a configured constant preserves image shape
        # while matching training-time augmentation choices.
        value = config["background_fill"]
        steps.append(lambda img, mask: fill_background(img, mask, value))

    if config.get("concat_mask_channels", False):
        # Concatenating the mask as an extra channel exposes anatomy location to
        # models trained with four-channel inputs.
        steps.append(lambda img, mask: concat_channels(img, mask))

    return steps


def run_pipeline(img, mask, steps: List[Callable]) -> np.ndarray:
    """Apply metadata-defined preprocessing steps to an image and mask.

    Args:
        img: Image array with shape ``(H, W, C)`` or ``(1, H, W, C)``.
        mask: Mask array with shape ``(H, W)``, ``(H, W, 1)``, or batched
            equivalents.
        steps: Pipeline callables produced by ``build_pipeline``.

    Returns:
        Batched image array ready for a classifier or segmentation model.
    """
    for step in steps:
        # Steps may update only the image or return a new image/mask pair when
        # later transforms need the adjusted mask as well.
        result = step(img, mask)
        if isinstance(result, tuple):
            img, mask = result
        else:
            img = result
    return ensure_batch(img)
