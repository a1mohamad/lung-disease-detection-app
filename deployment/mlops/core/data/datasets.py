"""TFRecord dataset builders for MLOps evaluation and retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import tensorflow as tf

from mlops.core.features.preprocess import apply_preprocess_config, make_parse_fn, remap_for_binary


def _create_parsed_dataset(
    tfrecord_paths: Iterable[Path],
    image_size: tuple[int, int],
) -> tf.data.Dataset:
    """Create a parsed dataset from TFRecord paths using configured image sizes.

    Args:
        tfrecord_paths: TFRecord shard paths for one split.
        image_size: Output image and mask size in ``(height, width)`` order.

    Returns:
        Unbatched dataset yielding parsed ``(image, mask, label)`` tensors.
    """
    # Parsing is kept separate from task-specific mapping so binary,
    # multiclass, and segmentation datasets share one TFRecord contract.
    parse_fn = make_parse_fn(image_size=image_size, mask_size=image_size)
    dataset = tf.data.TFRecordDataset([str(p) for p in tfrecord_paths])
    return dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _apply_preprocess_if_configured(dataset: tf.data.Dataset, preprocess_config: dict | None) -> tf.data.Dataset:
    """Apply metadata-driven preprocessing to parsed image, mask, label rows.

    Args:
        dataset: Parsed dataset yielding image, mask, and label tensors.
        preprocess_config: Optional metadata preprocessing block.

    Returns:
        Dataset with transformed image/mask tensors and original labels.
    """
    if not preprocess_config:
        return dataset
    # The preprocessing helper returns image and mask only; labels are threaded
    # through unchanged for supervised tasks.
    return dataset.map(
        lambda image, mask, label: (*apply_preprocess_config(image, mask, preprocess_config), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _finalize_classification_dataset(
    dataset: tf.data.Dataset,
    batch_size: int,
    preprocess_input: Callable[[tf.Tensor], tf.Tensor],
    preprocess_config: dict | None,
) -> tf.data.Dataset:
    """Batch, preprocess, and prefetch a classification dataset.

    Args:
        dataset: Dataset yielding classification rows.
        batch_size: Number of samples per batch.
        preprocess_input: Keras application preprocessing function.
        preprocess_config: Optional metadata preprocessing block.

    Returns:
        Prefetched dataset yielding ``(images, labels)`` batches.
    """
    dataset = dataset.batch(batch_size, drop_remainder=False)
    use_preprocess_input = not (preprocess_config and preprocess_config.get("preprocess_input_fn"))

    def _prep(*batch):
        """Normalize batched classifier rows into ``(images, labels)`` pairs.

        Args:
            *batch: Batched tensors, either ``(images, masks, labels)`` before
                task mapping or ``(images, labels)`` after task mapping.

        Returns:
            Tuple of image batch and label batch.

        Raises:
            ValueError: If the batch structure is not supported.
        """
        if len(batch) == 3:
            images, _masks, labels = batch
        elif len(batch) == 2:
            images, labels = batch
        else:
            raise ValueError(f"Unexpected classification batch format with {len(batch)} tensors")
        if use_preprocess_input:
            # Avoid double-preprocessing when metadata already declared the
            # model-specific preprocessing function.
            return preprocess_input(images), labels
        return images, labels

    dataset = dataset.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_binary_dataset(
    tfrecord_paths: Iterable[Path],
    image_size: tuple[int, int],
    batch_size: int,
    preprocess_input: Callable[[tf.Tensor], tf.Tensor],
    preprocess_config: dict | None = None,
) -> tf.data.Dataset:
    """Build a healthy/unhealthy classification dataset from TFRecords.

    Raw dataset labels use the original four-class research mapping. This
    builder remaps ``Normal`` to healthy and every disease class to unhealthy,
    then applies model-specific preprocessing before batching and prefetching.
    """
    dataset = _create_parsed_dataset(tfrecord_paths, image_size)
    # Raw labels come from the research dataset; the binary production task
    # collapses all abnormal classes into the unhealthy target.
    dataset = dataset.map(remap_for_binary, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = _apply_preprocess_if_configured(dataset, preprocess_config)
    return _finalize_classification_dataset(dataset, batch_size, preprocess_input, preprocess_config)


def build_multiclass_dataset(
    tfrecord_paths: Iterable[Path],
    image_size: tuple[int, int],
    batch_size: int,
    preprocess_input: Callable[[tf.Tensor], tf.Tensor],
    num_classes: int,
    preprocess_config: dict | None = None,
) -> tf.data.Dataset:
    """Build an unhealthy-only disease classification dataset from TFRecords.

    The disease classifier is trained only on abnormal classes, so normal
    samples are filtered out and the remaining labels are remapped to contiguous
    multiclass targets.
    """
    dataset = _create_parsed_dataset(tfrecord_paths, image_size)
    # Normal scans are removed before label remapping because the subtype model
    # should only learn disease categories.
    dataset = dataset.filter(lambda image, mask, label: tf.not_equal(label, 1))
    dataset = _apply_preprocess_if_configured(dataset, preprocess_config)

    def remap(image, mask, label):
        """Map raw disease labels to contiguous one-hot multiclass labels.

        Args:
            image: Parsed image tensor.
            mask: Parsed mask tensor, unused after filtering normal samples.
            label: Raw dataset class id.

        Returns:
            Tuple of image tensor and one-hot disease label.
        """
        keys = tf.constant([0, 2, 3], dtype=tf.int32)
        values = tf.constant(list(range(num_classes)), dtype=tf.int32)
        # A lookup table keeps the mapping inside the TensorFlow graph, which is
        # important for distributed tf.data execution.
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values),
            default_value=-1,
        )
        new_label = table.lookup(label)
        return image, tf.one_hot(new_label, depth=num_classes)

    dataset = dataset.map(remap, num_parallel_calls=tf.data.AUTOTUNE)
    return _finalize_classification_dataset(dataset, batch_size, preprocess_input, preprocess_config)


def build_segmentation_dataset(
    tfrecord_paths: Iterable[Path],
    image_size: tuple[int, int],
    batch_size: int,
    preprocess_config: dict | None = None,
) -> tf.data.Dataset:
    """Build a segmentation dataset of image and mask pairs from TFRecords.

    Classification labels are discarded after parsing because segmentation uses
    only the image and mask tensors. Optional preprocessing is still honored so
    evaluation and retraining match the model metadata.
    """
    dataset = _create_parsed_dataset(tfrecord_paths, image_size)

    def to_segmentation_pair(image, mask, label):
        """Drop class labels and return the image-mask pair used by segmentation.

        Args:
            image: Parsed image tensor.
            mask: Parsed segmentation mask.
            label: Raw class label, unused for segmentation.

        Returns:
            Tuple of image and mask tensors.
        """
        if preprocess_config:
            # Segmentation artifacts can still declare normalization or masking
            # transforms, so evaluation matches the saved model metadata.
            image, mask = apply_preprocess_config(image, mask, preprocess_config)
        return image, mask

    dataset = dataset.map(to_segmentation_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)
