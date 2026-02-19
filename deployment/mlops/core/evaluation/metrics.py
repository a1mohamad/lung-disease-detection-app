from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import tensorflow as tf


def eval_binary(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: Optional[int],
) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    auc = tf.keras.metrics.AUC()
    total = 0

    iterable: Iterable = dataset.take(max_batches) if max_batches else dataset
    for images, labels in iterable:
        preds = model(images, training=False)
        if preds.shape[-1] == 1:
            probs = tf.squeeze(tf.cast(preds, tf.float32), axis=-1)
        else:
            probs = tf.nn.softmax(tf.cast(preds, tf.float32), axis=-1)[:, 1]

        y_true = tf.cast(tf.squeeze(labels, axis=-1), tf.float32)
        y_pred = tf.cast(probs >= 0.5, tf.float32)

        tp += int(tf.reduce_sum(tf.cast((y_true == 1.0) & (y_pred == 1.0), tf.int32)))
        fp += int(tf.reduce_sum(tf.cast((y_true == 0.0) & (y_pred == 1.0), tf.int32)))
        fn += int(tf.reduce_sum(tf.cast((y_true == 1.0) & (y_pred == 0.0), tf.int32)))
        tn += int(tf.reduce_sum(tf.cast((y_true == 0.0) & (y_pred == 0.0), tf.int32)))

        auc.update_state(y_true, probs)
        total += int(labels.shape[0])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "samples": float(total),
        "val_accuracy": float(accuracy),
        "val_precision": float(precision),
        "val_recall": float(recall),
        "val_f1": float(f1),
        "val_auc": float(auc.result().numpy()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc.result().numpy()),
    }


def eval_multiclass(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: Optional[int],
    num_classes: int,
) -> Dict[str, float]:
    conf = tf.zeros((num_classes, num_classes), dtype=tf.int32)
    total = 0

    iterable: Iterable = dataset.take(max_batches) if max_batches else dataset
    for images, labels in iterable:
        preds = model(images, training=False)
        y_true = tf.argmax(labels, axis=-1, output_type=tf.int32)
        y_pred = tf.argmax(preds, axis=-1, output_type=tf.int32)
        conf += tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes, dtype=tf.int32)
        total += int(labels.shape[0])

    conf_f = tf.cast(conf, tf.float32)
    tp = tf.linalg.diag_part(conf_f)
    fp = tf.reduce_sum(conf_f, axis=0) - tp
    fn = tf.reduce_sum(conf_f, axis=1) - tp

    precision_per_class = tf.math.divide_no_nan(tp, tp + fp)
    recall_per_class = tf.math.divide_no_nan(tp, tp + fn)
    f1_per_class = tf.math.divide_no_nan(2.0 * precision_per_class * recall_per_class, precision_per_class + recall_per_class)

    accuracy = tf.math.divide_no_nan(tf.reduce_sum(tp), tf.cast(total, tf.float32))
    precision_macro = tf.reduce_mean(precision_per_class) if num_classes else tf.constant(0.0, dtype=tf.float32)
    recall_macro = tf.reduce_mean(recall_per_class) if num_classes else tf.constant(0.0, dtype=tf.float32)
    f1_macro = tf.reduce_mean(f1_per_class) if num_classes else tf.constant(0.0, dtype=tf.float32)

    return {
        "samples": float(total),
        "val_accuracy": float(accuracy.numpy()),
        "val_precision_macro": float(precision_macro.numpy()),
        "val_recall_macro": float(recall_macro.numpy()),
        "val_f1": float(f1_macro.numpy()),
        "val_f1_macro": float(f1_macro.numpy()),
        "accuracy": float(accuracy.numpy()),
        "precision_macro": float(precision_macro.numpy()),
        "recall_macro": float(recall_macro.numpy()),
    }


def eval_segmentation(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    max_batches: Optional[int],
) -> Dict[str, float]:
    dice_scores = []
    iou_scores = []
    total = 0
    eps = 1e-6

    iterable: Iterable = dataset.take(max_batches) if max_batches else dataset
    for images, masks in iterable:
        preds = model(images, training=False)
        if preds.shape[-1] == 1:
            probs = tf.sigmoid(preds)
        else:
            probs = preds
        pred_bin = tf.cast(probs >= 0.5, tf.float32)
        mask_bin = tf.cast(masks >= 0.5, tf.float32)

        intersection = tf.reduce_sum(pred_bin * mask_bin, axis=[1, 2, 3])
        union = tf.reduce_sum(pred_bin + mask_bin, axis=[1, 2, 3])
        dice = (2 * intersection + eps) / (union + eps)
        iou = (intersection + eps) / (union - intersection + eps)

        dice_scores.extend(dice.numpy().tolist())
        iou_scores.extend(iou.numpy().tolist())
        total += masks.shape[0]

    return {
        "samples": float(total),
        "val_dice_coefficient": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "val_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "dice_coefficient": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
    }
