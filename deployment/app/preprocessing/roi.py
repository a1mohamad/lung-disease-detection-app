import tensorflow as tf
from app.preprocessing.mask import binary_mask_to_rgb_batch
from app.preprocessing.transforms import ensure_batch

def crop_lung_roi(
    img: tf.Tensor,
    mask: tf.Tensor,
    target_size: tuple,
    threshold: float = 0.5,
    margin_ratio: float = 0.1,
) -> tf.Tensor:
    """
    Crop lung region using a binary segmentation mask and resize.

    Args:
        img: tf.Tensor, shape (1, H, W, 3) or (H, W, 3)
        mask: tf.Tensor, shape (H, W), (H, W, 1), or (1, H, W, 1)
        target_size: tuple (H, W)
        threshold: float, binarization threshold for mask
        margin_ratio: float, extra margin around lung bbox

    Returns:
        tf.Tensor: Cropped and resized image, shape (target_H, target_W, 3)
    """

    # --- Ensure proper shapes using transforms ---
    img = ensure_batch(img)       # shape: (1, H, W, C)
    mask_rgb = binary_mask_to_rgb_batch(mask)  # (1, H, W, 3)

    # --- Binarize mask ---
    mask_2d = tf.cast(mask_rgb[0, :, :, 0] > threshold, tf.float32)  # drop batch for indexing
    indices = tf.where(mask_2d > 0)
    img_shape = tf.shape(img)[1:3]  # H, W

    def crop():
        min_coords = tf.cast(tf.reduce_min(indices, axis=0), tf.int32)
        max_coords = tf.cast(tf.reduce_max(indices, axis=0), tf.int32)

        y_min, x_min = min_coords[0], min_coords[1]
        y_max, x_max = max_coords[0], max_coords[1]

        h = tf.cast(y_max - y_min, tf.int32)
        w = tf.cast(x_max - x_min, tf.int32)

        h_f = tf.cast(h, tf.float32)
        w_f = tf.cast(w, tf.float32)

        margin_y = tf.cast(tf.maximum(h_f * margin_ratio, 1.0), tf.int32)
        margin_x = tf.cast(tf.maximum(w_f * margin_ratio, 1.0), tf.int32)

        y_start = tf.maximum(0, y_min - margin_y)
        x_start = tf.maximum(0, x_min - margin_x)
        y_end = tf.minimum(img_shape[0], y_max + margin_y)
        x_end = tf.minimum(img_shape[1], x_max + margin_x)

        cropped = tf.image.crop_to_bounding_box(
            img[0], y_start, x_start, y_end - y_start, x_end - x_start
        )
        return tf.image.resize(cropped, target_size)

    def fallback():
        return tf.image.resize(img[0], target_size)

    # return final crop
    return tf.cond(tf.shape(indices)[0] > 0, crop, fallback)
