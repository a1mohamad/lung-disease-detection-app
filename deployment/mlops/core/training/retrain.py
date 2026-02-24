from __future__ import annotations

import tensorflow as tf

from mlops.core.data.datasets import (
    build_binary_dataset,
    build_multiclass_dataset,
    build_segmentation_dataset,
)
from mlops.core.data.tfrecord_ops import compute_steps_from_tfrecords
from mlops.core.evaluation.runner import evaluate_model_for_spec
from mlops.core.models.compile import get_preprocess_fn


def build_train_val_datasets_for_spec(spec, metadata: dict, train_files, val_files, batch_size: int):
    image_size = tuple(metadata.get("inference", {}).get("input_size", [256, 256]))
    preprocess_config = metadata.get("preprocessing", {})

    if spec.task == "binary_classification":
        prep = preprocess_config.get("preprocess_input_fn", "")
        preprocess_fn = get_preprocess_fn(prep)
        train_ds = build_binary_dataset(train_files, image_size, batch_size, preprocess_fn, preprocess_config)
        val_ds = build_binary_dataset(val_files, image_size, batch_size, preprocess_fn, preprocess_config)
        return train_ds, val_ds

    if spec.task == "multiclass_classification":
        prep = preprocess_config.get("preprocess_input_fn", "")
        preprocess_fn = get_preprocess_fn(prep)
        num_classes = len(metadata.get("output", {}).get("classes", {}))
        train_ds = build_multiclass_dataset(
            train_files,
            image_size,
            batch_size,
            preprocess_fn,
            num_classes,
            preprocess_config,
        )
        val_ds = build_multiclass_dataset(
            val_files,
            image_size,
            batch_size,
            preprocess_fn,
            num_classes,
            preprocess_config,
        )
        return train_ds, val_ds

    train_ds = build_segmentation_dataset(train_files, image_size, batch_size, preprocess_config)
    val_ds = build_segmentation_dataset(val_files, image_size, batch_size, preprocess_config)
    return train_ds, val_ds


def retrain_and_evaluate_for_spec(
    *,
    spec,
    metadata: dict,
    model: tf.keras.Model,
    train_files,
    val_files,
    batch_size: int,
    epochs: int,
    max_train_batches: int | None,
    max_eval_batches: int | None,
):
    train_steps = compute_steps_from_tfrecords(list(train_files), batch_size)
    val_steps = compute_steps_from_tfrecords(list(val_files), batch_size)

    if max_train_batches:
        train_steps = min(train_steps, max_train_batches)
    if max_eval_batches:
        val_steps = min(val_steps, max_eval_batches)

    train_ds, val_ds = build_train_val_datasets_for_spec(spec, metadata, train_files, val_files, batch_size)
    if max_train_batches:
        train_ds = train_ds.take(max_train_batches)
    if max_eval_batches:
        val_ds = val_ds.take(max_eval_batches)

    if train_steps <= 0:
        raise RuntimeError("Training steps resolved to zero. Check TFRecord files and batch size.")
    if val_steps <= 0:
        raise RuntimeError("Validation steps resolved to zero. Check TFRecord files and batch size.")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        train_ds,
        epochs=epochs,
        verbose=2,
        steps_per_epoch=train_steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
    )
    metrics = evaluate_model_for_spec(spec, model, val_ds, max_eval_batches, metadata)
    return model, metrics
