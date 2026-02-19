from keras import backend as K
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6
) -> tf.Tensor:
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
