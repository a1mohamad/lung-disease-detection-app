from __future__ import annotations

import tensorflow as tf

from mlops.core.data.datasets import (
    build_binary_dataset,
    build_multiclass_dataset,
    build_segmentation_dataset,
)
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
    train_ds, val_ds = build_train_val_datasets_for_spec(spec, metadata, train_files, val_files, batch_size)
    if max_train_batches:
        train_ds = train_ds.take(max_train_batches)

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
        verbose=1,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    metrics = evaluate_model_for_spec(spec, model, val_ds, max_eval_batches, metadata)
    return model, metrics
