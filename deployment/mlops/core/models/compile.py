"""Model compilation helpers used by evaluation and retraining jobs."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from app.utils.metrics import dice_coefficient


def get_preprocess_fn(name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the Keras preprocessing function identified by metadata.

    Args:
        name: Metadata preprocessing key such as ``densenet`` or ``inception``.

    Returns:
        Callable preprocessing function. Unknown names intentionally resolve to
        identity preprocessing so older metadata can still be evaluated.
    """
    name = name.lower()
    if name == "densenet":
        # Imports stay local so DAG parsing and non-Keras utilities do not pay
        # for every application module up front.
        from keras.applications.densenet import preprocess_input
    elif name == "efficientnet":
        from keras.applications.efficientnet_v2 import preprocess_input
    elif name == "inception":
        from keras.applications.inception_v3 import preprocess_input
    elif name == "mobilenet":
        from keras.applications.mobilenet_v3 import preprocess_input
    else:
        def preprocess_input(x):
            """Return tensors unchanged when no named preprocessing is required.

            Args:
                x: Tensor batch.

            Returns:
                The same tensor batch.
            """
            return x
    return preprocess_input


def compile_for_task(model: tf.keras.Model, task: str) -> tf.keras.Model:
    """Compile a model with task-appropriate loss and metrics.

    Args:
        model: Keras model to compile.
        task: Managed model task family.

    Returns:
        The same model instance after compilation.
    """
    if task == "binary_classification":
        # Binary models optimize the healthy/unhealthy screening decision.
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    elif task == "multiclass_classification":
        # Disease subtype models use one-hot labels after normal scans are
        # filtered out of the dataset builder.
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        # Segmentation models are evaluated by overlap quality, so Dice is the
        # tracked metric and binary crossentropy remains the pixel-wise loss.
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [dice_coefficient]

    # A conservative optimizer default keeps retraining jobs stable when model
    # artifacts are fine-tuned from existing weights.
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=metrics)
    return model


def load_model_local(path: str, task: str) -> tf.keras.Model:
    """Load a local Keras model, including custom segmentation metrics when needed.

    Args:
        path: Local Keras model path.
        task: Model task used to decide whether custom objects are required.

    Returns:
        Loaded Keras model.
    """
    if task == "segmentation":
        # Segmentation artifacts were saved with a custom Dice metric; Keras
        # needs the symbol when deserializing the model.
        return tf.keras.models.load_model(path, custom_objects={"dice_coefficient": dice_coefficient})
    return tf.keras.models.load_model(path)
