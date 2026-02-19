from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import tensorflow as tf

from mlops.core.features.preprocess import apply_preprocess_config, make_parse_fn, remap_for_binary


def _create_parsed_dataset(
    tfrecord_paths: Iterable[Path],
    image_size: tuple[int, int],
) -> tf.data.Dataset:
    parse_fn = make_parse_fn(image_size=image_size, mask_size=image_size)
    dataset = tf.data.TFRecordDataset([str(p) for p in tfrecord_paths])
    return dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _apply_preprocess_if_configured(dataset: tf.data.Dataset, preprocess_config: dict | None) -> tf.data.Dataset:
    if not preprocess_config:
        return dataset
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
    dataset = dataset.batch(batch_size, drop_remainder=False)
    use_preprocess_input = not (preprocess_config and preprocess_config.get("preprocess_input_fn"))

    def _prep(images, labels):
        if use_preprocess_input:
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
    dataset = _create_parsed_dataset(tfrecord_paths, image_size)
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
    dataset = _create_parsed_dataset(tfrecord_paths, image_size)
    dataset = dataset.filter(lambda image, mask, label: tf.not_equal(label, 1))
    dataset = _apply_preprocess_if_configured(dataset, preprocess_config)

    def remap(image, mask, label):
        keys = tf.constant([0, 2, 3], dtype=tf.int32)
        values = tf.constant(list(range(num_classes)), dtype=tf.int32)
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
    dataset = _create_parsed_dataset(tfrecord_paths, image_size)

    def to_segmentation_pair(image, mask, label):
        if preprocess_config:
            image, mask = apply_preprocess_config(image, mask, preprocess_config)
        return image, mask

    dataset = dataset.map(to_segmentation_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)

