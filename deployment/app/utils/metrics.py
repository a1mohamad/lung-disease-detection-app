import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def dice_coefficient(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6
) -> tf.Tensor:
    # Use TensorFlow ops directly for compatibility with Keras/TF versions
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
