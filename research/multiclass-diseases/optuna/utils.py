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
import tensorflow.keras.layers as tfl
from tensorflow.keras import metrics



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


def make_parse_fn(config):
    """
    Factory that creates a TFRecord parsing function with fixed image sizes.

    Args:
        model config containing:
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
        image_size = config['img_size']
        mask_size = config['mask_size']
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


def count_steps_from_dataset(dataset):
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

'''
Professional ROI Crop instead of Hard Masking
10% margin preserved for anatomical context.
'''
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


def get_dataset_metadata(dataset, batch_size):
    '''
    Professional one-pass logic to get counts, weights, and steps simultaneously.
    '''
    # 1. Initialize a zero tensor for the 3 classes
    initial_state = tf.zeros((3,), dtype=tf.float32)
    
    # 2. Single pass: Sum up all one-hot labels
    final_counts = dataset.reduce(
        initial_state, 
        lambda x, data: x + tf.reduce_sum(data[1], axis=0)
    ).numpy()
    
    # 3. Derive everything from these counts
    total_samples = int(sum(final_counts))
    steps_per_epoch = total_samples // batch_size
    
    # 4. Calculate Weights
    # Formula: Total / (NumClasses * ClassCount)
    num_classes = len(final_counts)
    class_weights = {
        i: total_samples / (num_classes * final_counts[i]) 
        for i in range(num_classes)
    }
    
    return {
        "steps": steps_per_epoch,
        "weights": class_weights,
        "counts": final_counts
    }


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


def make_remap_for_multiclass(num_classes):
    def remap_for_multiclass(image, label):
        KEYS = tf.constant([0, 2, 3], dtype= tf.int32)
        VALUES = tf.constant([0, 1, 2], dtype= tf.int32)
        TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(KEYS, VALUES),
        default_value= -1
        )
        new_label = TABLE.lookup(label)
    
        return image, tf.one_hot(new_label, depth= num_classes)
    return remap_for_multiclass

def penalized_f1_score(history, config, mode=None, loss=False):

    
    """
    Your exact rolling window penalized F1 score function
    """
    alpha_p = config['alpha_p']
    stage_epochs = config['stage']

    
    val_f1 = np.array(history.history["val_f1_score"])
    val_prec = np.array(history.history["val_precision"])
    val_rec = np.array(history.history["val_recall"])
    
    # Use last N epochs (adaptive for short architecture search)
    stage_epochs = min(stage_epochs, len(val_f1))
    
    if mode == 'roll':
        # Rolling window average
        K = config['K']
        f1_roll = np.convolve(val_f1[-stage_epochs:], np.ones(K)/K, mode="valid")
        prec_roll = np.convolve(val_prec[-stage_epochs:], np.ones(K)/K, mode="valid")
        rec_roll = np.convolve(val_rec[-stage_epochs:], np.ones(K)/K, mode="valid")
        
        # Best epoch by F1
        idx = np.argmax(f1_roll)
        
        f1 = f1_roll[idx]
        prec = prec_roll[idx]
        rec = rec_roll[idx]
    elif mode == 'mean':
        f1 = np.mean(val_f1[-stage_epochs:])
        prec = np.mean(val_prec[-stage_epochs:])
        rec = np.mean(val_rec[-stage_epochs:])
    else:
        print("Unknown Mode!")

    loss_penalty = 0
    if loss:
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        loss_penalty = alpha_p * (val_loss - train_loss)
        
    # Your gap penalty
    gap_penalty = alpha_p * abs(prec - rec)
    score = f1 - gap_penalty - loss_penalty
    
    return score, f1, prec, rec

def unfreeze_backbone(model, backbone_name= None, unfreeze_layer= None):
    base_model = model.get_layer(backbone_name)
    
    if unfreeze_layer is None:
        # Stage 1: Freeze everything
        base_model.trainable = False
        return model

    # Stage 2: Selective Unfreezing
    base_model.trainable = True
    unfreeze_flag = False
    
    for layer in base_model.layers:
        if layer.name == unfreeze_layer:
            unfreeze_flag = True
        
        if unfreeze_flag:
            # PROFESSIONAL RULE: Always keep BatchNormalization frozen during fine-tuning
            # to avoid destroying the moving mean/variance statistics.
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        else:
            layer.trainable = False
            
    return model

def compile_model(model, loss, optimizer):
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.metrics.F1Score(name="f1_score", average="macro"),
            tf.metrics.AUC(name='AUC')
        ],
    )
    return model

def multiclass_dataset(tfrecords, config, is_training= True, image_augmentation=None):
    shuffle_size = config["shuffle"]
    batch_size = config["batch_size"]
    AUTO = config["auto"]
    parse_fn = config["parse_fn"]
    remap_for_multiclass = config["remap"]
    long_roi_preprocess = config["roi"]
    preprocess_input = config["preprocess"]
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads= AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(parse_fn, num_parallel_calls= AUTO)
    dataset = dataset.filter(lambda image, mask, label: tf.not_equal(label, 1))
    dataset = dataset.map(lung_roi_preprocess, num_parallel_calls=AUTO)
    dataset = dataset.map(remap_for_multiclass, num_parallel_calls=AUTO)
    
    if is_training:
        dataset = dataset.shuffle(shuffle_size)
        # 1. Batch the data FIRST
        dataset = dataset.batch(batch_size, drop_remainder= True)
        # 2. Apply augmentation to the entire batch SECOND
        if image_augmentation is not None:
            dataset = dataset.map(
                lambda x, y: (image_augmentation(x, training=True), y), 
                num_parallel_calls= AUTO
            )
    else:
        # For validation, just batch the data without augmenting
        dataset = dataset.batch(batch_size, drop_remainder= True)

    dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls= AUTO)
    # 3. Prefetch the augmented batches
    dataset = dataset.prefetch(AUTO)
    return dataset

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

def densenet_model(
    hparams, dropout_rate,
    config=None, phase=None
):
    
    img_size = config["img_size"]
    num_classes = config["num_classes"]
    inputs = tfl.Input(shape= img_size + (3,))
    base_model = tf.keras.applications.DenseNet121(
        name= 'densenet',
        weights= 'imagenet',
        include_top= False
    )
    base_model.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    
    densenet = base_model(inputs, training= False)
    x = tfl.GlobalAveragePooling2D()(densenet)
    num_dense_layers = hparams["num_layers"]
    for i in range(num_dense_layers):
        units = hparams["dense_units"][i]
        x = tfl.Dense(units, activation= 'relu', name=f"head_dense_{i}")(x)
        if phase == 'arch':
            if i == num_dense_layers - 1:
                head_dropout = tfl.Dropout(dropout_rate, name="head_dropout")
                x = head_dropout(x, training=True)

        elif phase == 'opt':
            dropout = tfl.Dropout(dropout_rate, name=f"dropout_{i}")
            x = dropout(x, training=True)
            
        else:
            print(f"Unknown phase! arch or opt!")

    outputs = tfl.Dense(num_classes, activation= 'softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    return model