import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from app.preprocessing.transforms import normalize_image

def overlay_mask_on_image(
    original_img: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    thresh: float = 0.5,
) -> Image.Image:
    base = Image.fromarray(original_img.astype("uint8")).convert("RGBA")

    mask_2d = np.squeeze(mask)
    mask_bin = (mask_2d >= thresh).astype("uint8") * 255

    mask_l = Image.fromarray(mask_bin, mode="L").resize(base.size, Image.NEAREST)

    red = Image.new("RGBA", base.size, (255, 0, 0, int(255 * alpha)))
    transparent = Image.new("RGBA", base.size, (0, 0, 0, 0))

    overlay = Image.composite(red, transparent, mask_l)
    return Image.alpha_composite(base, overlay)


def plot_prediction_bars(
    probs_by_label: dict,
    title: str = "Prediction Probabilities",
) -> None:
    labels = list(probs_by_label.keys())
    values = [float(probs_by_label[k]) for k in labels]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualization(img: tf.Tensor, roi_img: tf.Tensor, mask: tf.Tensor) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].axis("off")
    img_n = normalize_image(tf.squeeze(img), mode='imagenet')
    axs[0].imshow(img_n)
    axs[0].set_title("Image")

    axs[1].axis("off")
    roi_img_n = normalize_image(tf.squeeze(roi_img), mode='imagenet')
    axs[1].imshow(roi_img_n)
    axs[1].set_title("Cropped ROI")

    axs[2].axis("off")
    axs[2].imshow(tf.squeeze(mask), cmap="gray")
    axs[2].set_title("Predicted Mask")

    plt.show()
