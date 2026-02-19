from __future__ import annotations

from typing import Callable

import tensorflow as tf

from app.utils.metrics import dice_coefficient


def get_preprocess_fn(name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    name = name.lower()
    if name == "densenet":
        from keras.applications.densenet import preprocess_input
    elif name == "efficientnet":
        from keras.applications.efficientnet_v2 import preprocess_input
    elif name == "inception":
        from keras.applications.inception_v3 import preprocess_input
    elif name == "mobilenet":
        from keras.applications.mobilenet_v3 import preprocess_input
    else:
        def preprocess_input(x):
            return x
    return preprocess_input


def compile_for_task(model: tf.keras.Model, task: str) -> tf.keras.Model:
    if task == "binary_classification":
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    elif task == "multiclass_classification":
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [dice_coefficient]

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=metrics)
    return model


def load_model_local(path: str, task: str) -> tf.keras.Model:
    if task == "segmentation":
        return tf.keras.models.load_model(path, custom_objects={"dice_coefficient": dice_coefficient})
    return tf.keras.models.load_model(path)
