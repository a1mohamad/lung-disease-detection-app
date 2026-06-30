"""TFRecord parsing and preprocessing features for MLOps datasets."""

from __future__ import annotations

import tensorflow as tf

from mlops.core.models.compile import get_preprocess_fn


def _ensure_channel(mask: tf.Tensor) -> tf.Tensor:
    """Ensure a mask tensor has an explicit channel dimension.

    Args:
        mask: Tensor shaped ``H x W`` or ``H x W x C``.

    Returns:
        Mask tensor with channel-last shape.
    """
    # Dataset masks can arrive as HxW or HxWx1 depending on the writer; model
    # pipelines consistently consume channel-last tensors.
    return mask if mask.shape.rank and mask.shape.rank >= 3 else tf.expand_dims(mask, axis=-1)


def _apply_mask(image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Apply a binary mask to an image tensor.

    Args:
        image: Image tensor.
        mask: Binary foreground mask.

    Returns:
        Image tensor with background pixels zeroed.
    """
    return image * mask


def _fill_background(image: tf.Tensor, mask: tf.Tensor, value: float) -> tf.Tensor:
    """Fill masked-out image regions with a constant value.

    Args:
        image: Image tensor.
        mask: Binary foreground mask.
        value: Fill value for masked-out background pixels.

    Returns:
        Image tensor with foreground preserved and background filled.
    """
    background = (1.0 - mask) * value
    return _apply_mask(image, mask) + background


def _concat_channels(image: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Append the mask as an additional image channel.

    Args:
        image: Image tensor in channel-last format.
        mask: Binary mask tensor.

    Returns:
        Tensor with mask concatenated on the channel axis.
    """
    mask = _ensure_channel(mask)
    return tf.concat([image, mask], axis=-1)


def _normalize_image(image: tf.Tensor, mode: str) -> tf.Tensor:
    """Normalize an image according to the named preprocessing mode.

    Args:
        image: Image tensor.
        mode: Metadata normalization mode such as ``imagenet``, ``[-1,1]``, or
            ``none``.

    Returns:
        Float32 image tensor normalized according to ``mode``.
    """
    image = tf.cast(image, tf.float32)
    if mode == "imagenet":
        return image / 255.0
    if mode == "[-1,1]":
        return (image / 127.5) - 1.0
    if mode == "none":
        return image
    return image


def crop_lung_roi_sample(
    image: tf.Tensor,
    mask: tf.Tensor,
    target_size: tuple[int, int],
    threshold: float = 0.5,
    margin_ratio: float = 0.1,
) -> tf.Tensor:
    """Crop one sample to the mask bounding box with a contextual margin.

    Args:
        image: Single image tensor.
        mask: Single binary lung mask tensor.
        target_size: Output size in ``(height, width)`` order.
        threshold: Mask threshold used to identify lung pixels.
        margin_ratio: Fractional context retained around the lung box.

    Returns:
        Resized ROI tensor, or the resized full image when the mask is empty.
    """
    mask = _ensure_channel(mask)
    # Threshold first so interpolation artifacts in masks do not expand the ROI
    # with low-confidence background pixels.
    mask_2d = tf.cast(mask[:, :, 0] > threshold, tf.float32)
    indices = tf.where(mask_2d > 0)
    img_shape = tf.shape(image)[:2]

    def crop():
        """Crop and resize the mask bounding box when foreground pixels exist.

        Returns:
            Resized lung ROI tensor.
        """
        min_coords = tf.cast(tf.reduce_min(indices, axis=0), tf.int32)
        max_coords = tf.cast(tf.reduce_max(indices, axis=0), tf.int32)

        y_min, x_min = min_coords[0], min_coords[1]
        y_max, x_max = max_coords[0], max_coords[1]

        # Margins retain contextual anatomy around the lung box, matching the
        # research preprocessing used to train ROI classifiers.
        h = tf.cast(y_max - y_min, tf.float32)
        w = tf.cast(x_max - x_min, tf.float32)

        margin_y = tf.cast(h * margin_ratio, tf.int32)
        margin_x = tf.cast(w * margin_ratio, tf.int32)

        y_start = tf.maximum(0, y_min - margin_y)
        x_start = tf.maximum(0, x_min - margin_x)
        y_end = tf.minimum(img_shape[0], y_max + margin_y)
        x_end = tf.minimum(img_shape[1], x_max + margin_x)

        final_h = tf.maximum(y_end - y_start, 1)
        final_w = tf.maximum(x_end - x_start, 1)

        # crop_to_bounding_box requires strictly positive sizes; the safeguards
        # above keep empty or razor-thin masks from failing the data pipeline.
        cropped = tf.image.crop_to_bounding_box(
            image, y_start, x_start, final_h, final_w
        )
        return tf.image.resize(cropped, target_size)

    def fallback():
        """Resize the full image when the mask has no foreground pixels.

        Returns:
            Resized full-image tensor.
        """
        return tf.image.resize(image, target_size)

    # Use tf.cond so both branches remain graph-compatible inside tf.data maps.
    return tf.cond(tf.shape(indices)[0] > 0, crop, fallback)


def apply_preprocess_config(
    image: tf.Tensor,
    mask: tf.Tensor,
    config: dict | None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply metadata-driven preprocessing to one image and mask sample.

    The same metadata concepts used in the API are mirrored here so training,
    evaluation, and serving transformations remain aligned across the project.
    """
    if not config:
        return image, mask

    if config.get("mask_as_rgb") and config.get("concat_mask_channels"):
        raise ValueError("mask_as_rgb and concat_mask_channels cannot both be true.")

    if config.get("use_roi"):
        # ROI cropping must happen before later pixel-level operations because
        # it changes image geometry.
        target_size = tuple(config.get("roi_target_size", (256, 256)))
        image = crop_lung_roi_sample(image, mask, target_size)

    if config.get("preprocess_input_fn"):
        # Metadata selects the same named preprocessing functions used by the
        # serving code, keeping train/eval/serve transforms aligned.
        preprocess = get_preprocess_fn(config["preprocess_input_fn"])
        image = preprocess(image)

    if config.get("normalize"):
        image = _normalize_image(image, config["normalize"])

    if config.get("use_mask"):
        # Applying the mask after normalization mirrors the historical research
        # pipelines for these artifacts.
        image = _apply_mask(image, mask)

    if config.get("background_fill") is not None:
        image = _fill_background(image, mask, float(config["background_fill"]))

    if config.get("concat_mask_channels"):
        image = _concat_channels(image, mask)

    return image, mask


def make_parse_fn(image_size, mask_size):
    """Create a parser for TFRecord examples with configured output sizes.

    Args:
        image_size: Target size for decoded RGB images.
        mask_size: Target size for decoded binary masks.

    Returns:
        Callable that parses serialized TFRecord examples into image, mask, and
        integer class label tensors.
    """
    def parse_fn(example):
        """Parse one serialized example into image, mask, and class label tensors.

        Args:
            example: Serialized TFRecord example.

        Returns:
            Tuple of decoded image tensor, binary mask tensor, and class label.
        """
        # TFRecord examples store PNG-encoded image and mask bytes plus the raw
        # dataset class id used by all downstream task builders.
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "class": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(example, feature_description)

        img = tf.io.decode_png(example["image"], channels=3)
        mask = tf.io.decode_png(example["mask"], channels=1)

        img = tf.image.resize(img, image_size, method="bilinear")
        img = tf.cast(img, tf.float32)

        # Nearest-neighbor resize preserves binary masks; bilinear interpolation
        # would introduce fractional class boundaries.
        mask = tf.image.resize(mask, mask_size, method="nearest")
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.round(mask)

        label = tf.cast(example["class"], tf.int32)
        return img, mask, label

    return parse_fn


def remap_for_binary(image, mask, label):
    """Convert the raw multiclass label into the binary healthy/unhealthy target.

    Args:
        image: Image tensor.
        mask: Mask tensor.
        label: Raw multiclass label where ``1`` represents Normal.

    Returns:
        Tuple of image, mask, and binary label tensor.
    """
    new_label = tf.where(tf.equal(label, 1), 0, 1)
    new_label = tf.cast(new_label, tf.float32)
    new_label = tf.expand_dims(new_label, axis=-1)
    return image, mask, new_label
