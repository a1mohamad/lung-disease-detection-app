
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


def remap_for_binary(image, mask, label):
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

    return image, mask, new_label


def make_augment(geometric_aug):
    """
    Factory that builds a synchronized image-mask augmentation function.

    Args:
        geometric_aug (tf.keras.layers.Layer): Augmentation layer applied
            jointly to image and mask.

    Returns:
        Callable: Augmentation function applied on batched data.
    """

    def augment(image, mask, label):
        """
        Apply geometric augmentation to image and mask,
        and color augmentation to image only.

        Args:
            image (tf.Tensor): Batch of images.
            mask (tf.Tensor): Batch of masks.
            label (tf.Tensor): Batch of labels.

        Returns:
            tuple: (augmented_image, augmented_mask, label)
        """
        # Concatenation ensures identical geometric transforms for image & mask
        img_mask_concat = tf.concat([image, mask], axis=-1)
        img_mask_concat = geometric_aug(img_mask_concat, training=True)

        image = img_mask_concat[..., :3]
        mask = tf.round(img_mask_concat[..., 3:])  # Restore binary mask

        # Color augmentation is applied only to the image
        image = tf.image.random_contrast(image, 0.95, 1.05)
        image = tf.image.random_brightness(image, 0.95, 1.05)
        image = tf.clip_by_value(image, 0.0, 255.0)

        return image, mask, label

    return augment


def build_binary_dataset(
    tfrecords,
    parse_fn,
    augment,
    remap_fn,
    preprocess,
    shuffle_size,
    batch_size,
    is_training=True,
):
    """
    Build a high-performance tf.data pipeline for binary image classification.

    This pipeline:
    - Reads TFRecord files with parallel I/O
    - Parses and decodes image, mask, and label data
    - Remaps original labels into a binary classification target
    - Applies optional shuffling and batch-level augmentation during training
    - Applies final model-specific preprocessing
    - Prefetches batches for efficient CPU/GPU/TPU utilization

    Args:
        tfrecords (list[str]): Paths to TFRecord files.
        parse_fn (Callable): Function that parses a serialized TFRecord example
            into (image, mask, label).
        augment (Callable): Batch-level augmentation function applied only
            during training.
        remap_fn (Callable): Function that remaps raw labels into binary targets.
        preprocess (Callable): Final preprocessing function applied after
            batching (e.g., masking, normalization).
        shuffle_size (int): Buffer size used for shuffling training data.
        batch_size (int): Number of samples per batch.
        is_training (bool): Whether to enable training-specific behavior
            (shuffling and augmentation).

    Returns:
        tf.data.Dataset: A prepared, prefetched dataset ready for model training
    """
    options = tf.data.Options()
    options.experimental_deterministic = False  # Improves input throughput

    dataset = tf.data.TFRecordDataset(
        tfrecords, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(options)

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(remap_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def count_steps_from_tfrecord(tfrecords, batch_size):
    """
    Compute the number of training steps per epoch from TFRecord files.

    This function iterates through all TFRecord files to count the total
    number of serialized examples and converts it into a step count
    based on the provided batch size.

    Args:
        tfrecords (list[str]): Paths to TFRecord files.
        batch_size (int): Batch size used during training.

    Returns:
        int: Number of steps per epoch.
    """
    count = 0
    for tfrecord in tfrecords:
        # Iterates once through TFRecord to count samples
        count += sum(1 for _ in tf.data.TFRecordDataset(tfrecord))

    return math.ceil(count / batch_size)


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


def build_multiclass_dataset(
    tfrecords,
    parse_fn,
    augment,
    remap_fn,
    preprocess,
    shuffle_size,
    batch_size,
    is_training=True,
):
    """
    Build a high-performance tf.data pipeline for multiclass image classification.

    This pipeline:
    - Reads TFRecord files with parallel I/O
    - Parses and decodes image, mask, and label data
    - Filters out the "normal" class (label == 1)
    - Remaps remaining labels to one-hot encoded multiclass targets
    - Applies optional shuffling and batch-level augmentation
    - Applies final model-specific preprocessing
    - Prefetches batches for efficient device utilization

    Args:
        tfrecords (list[str]): Paths to TFRecord files.
        parse_fn (Callable): Function that parses a serialized TFRecord example
            into (image, mask, label).
        augment (Callable): Batch-level augmentation function applied during
            training only.
        remap_fn (Callable): Function that remaps raw labels into multiclass
            one-hot encoded targets.
        preprocess (Callable): Final preprocessing function applied after
            batching (e.g., masking, normalization).
        shuffle_size (int): Buffer size used for shuffling training data.
        batch_size (int): Number of samples per batch.
        is_training (bool): Whether to enable training-specific behavior
            (shuffling and augmentation).

    Returns:
        tf.data.Dataset: A prepared, prefetched dataset ready for model training
        or evaluation.
    """
    options = tf.data.Options()
    options.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(
        tfrecords, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(options)

    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Explicitly remove normal class (label == 1) for multiclass setup
    dataset = dataset.filter(lambda image, mask, label: tf.not_equal(label, 1))

    dataset = dataset.map(remap_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(shuffle_size)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def lung_roi_preprocess(image, mask, label):
    """Crop a lung-centered ROI for multiclass disease experiments.

    The function uses the segmentation mask to find the lung bounding box,
    expands it with a small context margin, and resizes the crop to the model
    input size. If the mask is empty, the full image is resized so the dataset
    pipeline remains robust during exploratory experiments.
    """
    mask_2d = tf.cast(mask[:, :, 0], tf.float32)
    indices = tf.where(mask_2d > 0.5)

    img_shape = tf.shape(image)

    def crop():
        """Crop and resize the mask bounding box when lung pixels are present.

        Returns:
            Resized lung ROI tensor.
        """
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
        """Resize the full image when no mask foreground is available.

        Returns:
            Resized full-image tensor.
        """
        return tf.image.resize(image, (256, 256))

    image = tf.cond(tf.shape(indices)[0] > 0, crop, fallback)

    return image, label


def get_dataset_metadata(dataset, batch_size):
    """Compute class counts, class weights, and epoch steps in one pass.

    Args:
        dataset: Batched dataset yielding ``(image_batch, one_hot_labels)``.
        batch_size: Batch size used to convert sample counts into steps.

    Returns:
        Dictionary with ``steps``, ``weights``, and raw per-class ``counts``.
    """
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

def compute_per_class_metrics(model, dataset, class_names):
    """Evaluate precision, recall, and F1 for each class in a dataset.

    This helper is intended for notebook diagnostics where macro averages are
    not enough. It materializes predictions, compares them to one-hot labels,
    and returns one metric dictionary per class name.
    """
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        zero_division=0
    )

    results = {}
    for i, cls in enumerate(class_names):
        results[cls] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i]
        }

    return results


class PerClassMetricsCallback(tf.keras.callbacks.Callback):
    """Keras callback that prints per-class metrics after selected epochs.

    The callback is useful during notebook training runs because it surfaces
    class-specific collapse early, especially for imbalanced disease labels
    where aggregate validation accuracy can look deceptively healthy.
    """

    def __init__(self, val_dataset, class_names, every_n_epochs=1):
        """Store validation data, class labels, and reporting cadence.

        Args:
            val_dataset: Validation dataset used for per-class metrics.
            class_names: Ordered display names for classes.
            every_n_epochs: Metric reporting cadence.
        """
        super().__init__()
        self.val_dataset = val_dataset
        self.class_names = class_names
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        """Compute and print per-class metrics at the configured cadence.

        Args:
            epoch: Zero-based epoch index.
            logs: Optional Keras metric dictionary.
        """
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        results = compute_per_class_f1(
            self.model, self.val_dataset, self.class_names
        )

        print("\n📊 Per-class metrics:")
        for cls, m in results.items():
            print(
                f"{cls:15s} | "
                f"P: {m['precision']:.3f} "
                f"R: {m['recall']:.3f} "
                f"F1: {m['f1']:.3f}"
            )

def unfreeze_backbone(model, backbone_name= None, unfreeze_layer= None):
    """Freeze or selectively unfreeze a backbone for fine-tuning.

    Args:
        model: Keras model containing a named backbone layer.
        backbone_name: Name of the backbone layer to adjust.
        unfreeze_layer: First layer name to unfreeze. When omitted, the full
            backbone remains frozen.

    Returns:
        The same model instance with updated trainability flags.
    """
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

class StrictMetricsPruningCallback(tf.keras.callbacks.Callback):
    """Stop weak Optuna trials using F1 plus precision/recall constraints.

    Optuna pruning alone can keep trials that score well on F1 while collapsing
    precision or recall. This callback adds explicit clinical-style guardrails
    so optimization favors balanced classifiers instead of one-sided models.
    """

    def __init__(
        self,
        trial,
        monitor_f1="val_f1_score",
        monitor_prec="val_precision",
        monitor_rec="val_recall",
        min_prec=0.75,
        min_rec=0.75,
        start_checking_at=3,
        patience=1,
        verbose=True,
    ):
        """Configure monitored metrics, minimum constraints, and prune cadence.

        Args:
            trial: Optuna trial object.
            monitor_f1: Validation F1 metric name.
            monitor_prec: Validation precision metric name.
            monitor_rec: Validation recall metric name.
            min_prec: Minimum allowed precision after warmup.
            min_rec: Minimum allowed recall after warmup.
            start_checking_at: Epoch index before pruning checks begin.
            patience: Number of allowed constraint violations.
            verbose: Whether to print pruning messages.
        """
        super().__init__()
        self.trial = trial
        self.monitor_f1 = monitor_f1
        self.monitor_prec = monitor_prec
        self.monitor_rec = monitor_rec
        self.min_prec = min_prec
        self.min_rec = min_rec
        self.start_checking_at = start_checking_at
        self.patience = patience
        self.verbose = verbose

        self.wait = 0
        self.constraint_violation = False
        self.optuna_prune = False
        self.last_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        """Report F1 to Optuna and stop trials that violate quality thresholds.

        Args:
            epoch: Zero-based epoch index.
            logs: Optional Keras metric dictionary.
        """
        logs = logs or {}

        f1 = logs.get(self.monitor_f1)
        prec = logs.get(self.monitor_prec)
        rec = logs.get(self.monitor_rec)

        if f1 is None:
            return

        # Save last seen metrics (for final logging)
        self.last_metrics = {
            "epoch": epoch + 1,
            "f1": f1,
            "precision": prec,
            "recall": rec,
        }

        # ---- Report to Optuna ----
        self.trial.report(f1, step=epoch)

        # ---- Constraint check ----
        if epoch >= self.start_checking_at:
            if prec is not None and rec is not None:
                if prec < self.min_prec or rec < self.min_rec:
                    self.wait += 1

                    if self.verbose:
                        print(
                            f"\n⚠️ Trial {self.trial.number} | Epoch {epoch+1} | "
                            f"F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f} "
                            f"(below threshold {self.min_prec}) "
                            f"[patience {self.wait}/{self.patience}]"
                        )

                    if self.wait > self.patience:
                        self.constraint_violation = True
                        self.model.stop_training = True
                        return
                else:
                    self.wait = 0

        # ---- Optuna pruning ----
        if epoch >= self.start_checking_at and self.trial.should_prune():
            if self.verbose:
                print(
                    f"⏹️ Optuna prune requested | Trial {self.trial.number} | "
                    f"Epoch {epoch+1} | F1={f1:.4f}"
                )
            self.optuna_prune = True
            self.model.stop_training = True

def print_memory_usage():
    """Print the current process resident memory usage in megabytes.

    This is a notebook diagnostic helper for long Optuna sessions.
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"--- Current RAM Usage: {mem_mb:.2f} MB ---")


def cleanup(model, history, callbacks_list):
    """Release large training objects and clear the Keras backend session.

    This helper is designed for repeated notebook trials where stale graphs,
    histories, and callbacks can accumulate in memory between Optuna runs.
    """
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

def penalized_f1_score(history, config, mode=None, loss=False):

    
    """Score a trial by F1 while penalizing unstable validation behavior.

    The score rewards high validation F1 while subtracting penalties for
    precision/recall imbalance and, optionally, train-validation loss gaps.
    It is used as an Optuna objective component for more balanced candidates.
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


def compile_model(model, loss, optimizer):
    """Compile a multiclass model with accuracy, precision, recall, F1, and AUC.

    Args:
        model: Keras model to compile.
        loss: Keras loss function.
        optimizer: Keras optimizer.

    Returns:
        Compiled model instance.
    """
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

def densenet_model(
    hparams, dropout_rate,
    config=None, phase=None
):
    """Build a DenseNet121 classifier head for architecture searches.

    The ImageNet backbone stays frozen while Optuna varies dense-layer width,
    depth, and dropout behavior. Batch normalization layers remain frozen to
    preserve pretrained moving statistics during trial evaluation.
    """
    
    img_size = MODEL_CONFIG["img_size"]
    num_classes = MODEL_CONFIG["num_classes"]
    inputs = tfl.Input(shape= img_size + (3,))
    base_model = DenseNet121(
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

def multiclass_dataset(tfrecords, config, is_training= True, image_augmentation=None):
    """Build a multiclass disease-classification dataset from TFRecord files.

    Normal samples are removed, lung ROIs are cropped from masks, raw labels are
    remapped to contiguous disease classes, and optional batch augmentation is
    applied only during training.
    """
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

def make_remap_for_multiclass(num_classes):
    """Create a mapper that converts raw disease labels into one-hot targets.

    Args:
        num_classes: Number of disease classes after filtering Normal.

    Returns:
        Mapping function for ``tf.data.Dataset.map``.
    """
    def remap_for_multiclass(image, label):
        """Map COVID, Viral Pneumonia, and Lung Opacity to contiguous labels.

        Args:
            image: ROI image tensor.
            label: Raw dataset label.

        Returns:
            Tuple of image tensor and one-hot disease label.
        """
        KEYS = tf.constant([0, 2, 3], dtype= tf.int32)
        VALUES = tf.constant([0, 1, 2], dtype= tf.int32)
        TABLE = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(KEYS, VALUES),
        default_value= -1
        )
        new_label = TABLE.lookup(label)
    
        return image, tf.one_hot(new_label, depth= num_classes)
    return remap_for_multiclass


class OverfitCallback(tf.keras.callbacks.Callback):
    """Stop training when validation loss diverges from training loss.

    This notebook callback provides a simple guardrail for architecture trials
    where validation loss begins to drift away from training loss.
    """

    def __init__(self, 
                 overfit_threshold=OVERFIT_THRESHOLD,
                patience=2,
                verbose=True,
                ):
        """Configure the loss-gap threshold and patience for overfit detection.

        Args:
            overfit_threshold: Minimum validation-training loss gap to count.
            patience: Number of consecutive gap violations allowed.
            verbose: Whether to print stopping messages.
        """
        self.threshold = overfit_threshold
        self.patience = patience
        self.verbose = verbose

        self.wait = 0
        self.overfit_sign = False
    
    def on_epoch_end(self, epoch, logs=None):
        """Stop training when validation loss exceeds the configured gap.

        Args:
            epoch: Zero-based epoch index.
            logs: Keras metric dictionary for the epoch.
        """
        train_loss = logs['loss']
        val_loss = logs['val_loss']
        overfit_sign = val_loss - train_loss
        # Check the train and val loss
        if overfit_sign > self.threshold:
            self.wait += 1
            if self.verbose:
                print(
                    f"Sign of Overfitting!!!!\n"
                    f"Train Loss: {train_loss} | "
                    f"Val Loss:{val_loss} | "
                    f"[Patience {self.wait}/{self.patience}]"
                )
            if self.wait > self.patience:
                self.overfit_sign = True
        else:
            self.wait = 0

        if self.overfit_sign:
            if self.verbose:
                # Stop if threshold is met
                print("\noverfitting is happening!!! so cancelling training!")
                self.model.stop_training = True
                return
