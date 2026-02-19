# ------- Utils Python File -------

"""
Utility functions for reproducibility, data parsing, augmentation,
dataset construction, and training utilities.

This module is designed to keep the training notebooks clean by
centralizing reusable TensorFlow and data pipeline logic.
"""
import gc
import math
import numpy as np
import optuna
import os
import psutil
import random
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf


def seed_everthing(SEED=28):
    """
    Set global random seeds for reproducibility across TensorFlow, NumPy,
    and Python's random module.

    Args:
        SEED (int): Seed value used for all random number generators.
    """
    # Ensures reproducible behavior across runs (important for experiments)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print("For reproducibility, everything seeded!")


def gpu_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to be enabled for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return 


def get_strategy():
    """
    Detect and return the best available TensorFlow distribution strategy.

    Priority:
        1. TPU
        2. Multi-GPU
        3. CPU

    Returns:
        tf.distribute.Strategy: Initialized distribution strategy.
    """
    try:
        # Prefer TPU if available (fastest for large-scale training)
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Using TPU strategy:", type(strategy).__name__)
    except Exception:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Mirror model across all visible GPUs
            strategy = tf.distribute.MirroredStrategy()
            print("Using GPU strategy:", type(strategy).__name__)
        else:
            # Safe fallback for CPU-only environments
            strategy = tf.distribute.get_strategy()
            print("Using CPU strategy:", type(strategy).__name__)

    print("REPLICAS:", strategy.num_replicas_in_sync)
    return strategy


def make_parse_fn(image_size, mask_size):
    """
    Factory that creates a TFRecord parsing function with fixed image sizes.

    Args:
        image_size (tuple): Target (H, W) for image resizing.
        mask_size (tuple): Target (H, W) for mask resizing.

    Returns:
        Callable: A function that parses a single TFRecord example.
    """

    def parse_fn(example):
        """
        Parse and decode a single TFRecord example.

        Args:
            example (tf.Tensor): Serialized TFRecord example.

        Returns:
            tuple: (image, mask, label) tensors.
        """
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "class": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(example, feature_description)

        img = tf.io.decode_png(example["image"], channels=3)
        mask = tf.io.decode_png(example["mask"], channels=1)

        # Bilinear for images preserves visual quality
        img = tf.image.resize(img, image_size, method="bilinear")
        img = tf.cast(img, tf.float32)

        # Nearest neighbor avoids introducing soft edges in masks
        mask = tf.image.resize(mask, mask_size, method="nearest")
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.round(mask)  # Enforce strict binary mask

        label = tf.cast(example["class"], tf.int32)

        return img, mask, label

    return parse_fn



def lung_roi_preprocess(image, mask, label):
    mask_2d = tf.cast(mask[:, :, 0], tf.float32)
    indices = tf.where(mask_2d > 0.5)

    img_shape = tf.shape(image)

    def crop():
        min_coords = tf.cast(tf.reduce_min(indices, axis=0), tf.int32)
        max_coords = tf.cast(tf.reduce_max(indices, axis=0), tf.int32)

        y_min = min_coords[0]
        x_min = min_coords[1]
        y_max = max_coords[0]
        x_max = max_coords[1]

        h = tf.cast(y_max - y_min, tf.float32)
        w = tf.cast(x_max - x_min, tf.float32)

        margin_y = tf.cast(h * 0.1, tf.int32)
        margin_x = tf.cast(w * 0.1, tf.int32)

        y_start = tf.maximum(0, y_min - margin_y)
        x_start = tf.maximum(0, x_min - margin_x)
        y_end = tf.minimum(img_shape[0], y_max + margin_y)
        x_end = tf.minimum(img_shape[1], x_max + margin_x)

        final_h = tf.maximum(y_end - y_start, 1)
        final_w = tf.maximum(x_end - x_start, 1)

        cropped = tf.image.crop_to_bounding_box(
            image, y_start, x_start, final_h, final_w
        )
        return tf.image.resize(cropped, (256, 256))

    def fallback():
        return tf.image.resize(image, (256, 256))

    image = tf.cond(tf.shape(indices)[0] > 0, crop, fallback)

    return image, label


def remap_for_binary(image, label):
    """
    Convert a multiclass label into binary format.

    Label mapping:
        - Original label == 1 → 0
        - All other labels     → 1

    Args:
        image (tf.Tensor): Image tensor.
        mask (tf.Tensor): Mask tensor.
        label (tf.Tensor): Integer class label.

    Returns:
        tuple: (image, mask, binary_label) where label shape is [1].
    """
    # Collapse multiclass labels into a single positive/negative target
    new_label = tf.where(tf.equal(label, 1), 0, 1)
    new_label = tf.cast(new_label, tf.float32)
    new_label = tf.expand_dims(new_label, axis=-1)

    return image, new_label

def count_steps(dataset):
    """
    Count the number of batches yielded by a batched tf.data.Dataset.

    This function should be used only when the dataset is already batched.
    The returned value directly corresponds to the number of training or
    validation steps.

    Args:
        dataset (tf.data.Dataset): A batched dataset.

    Returns:
        int: Number of batches (steps).
    """
    # Uses dataset reduction instead of Python iteration (graph-safe)
    return dataset.reduce(
        tf.constant(0, tf.int64), lambda x, _: x + 1
    ).numpy()


def cleanup(model, history, callbacks_list):
    '''
    Explicitly clears the Keras session, deletes large objects, 
    and forces garbage collection to prevent RAM bloat.
    '''
    try:
        if history is not None:
            del history
        if callbacks_list is not None:
            del callbacks_list
        if model is not None:
            del model
    
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        gc.collect()

    
    # 4. Optional: Log RAM usage to verify cleanup
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"🧹 RAM Cleaned. Current usage: {mem_mb:.2f} MB")
