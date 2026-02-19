import os
from pathlib import Path

# Silence TensorFlow startup logs (must be set before importing TensorFlow)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from app.configs.config import AppConfig
from app.preprocessing.image import ImageLoadError, load_image
from app.predictor.pipeline import LungDetection
from app.utils.visualization import overlay_mask_on_image, visualization, plot_prediction_bars

import tensorflow as tf
tf.get_logger().setLevel("ERROR")


def main(image_path: str) -> None:
    img = load_image(image_path, target_size=AppConfig.IMAGE_SIZE)
    lung_detector = LungDetection()
    results = lung_detector.predict(img)
    orig = img.numpy()[0]          # already resized to 256x256
    mask = results["mask"].numpy()
    overlay = overlay_mask_on_image(orig, mask)
    overlay.save("overlay3.png")
    roi_img = results["roi_img"]
    mask = results["mask"]
    visualization(img, roi_img, mask)

    binary = results["binary"]
    file_name = Path(image_path).name
    print(f"Image: {file_name}")
    print(
        f"Final: label={binary['final_label']} "
        f"name={binary['final_label_name']} "
        f"Model raw Prob:{binary['final_prob']}"
    )
    final_probs = binary.get("final_probs_by_label", {})
    final_label = binary["final_label"]
    if final_label == 0:
        prob = final_probs.get("healthy", None)
        if prob is not None:
            print(f"Final (healthy prob): {prob:.4f}")
    else:
        prob = final_probs.get("unhealthy", None)
        if prob is not None:
            print(f"Final (unhealthy prob): {prob:.4f}")
    print("Per-model results:")
    for name, m in binary["models_results"].items():
        print(
            f"  {name}: label={m['label']} "
            f"name={m['label_name']} "
        )
        probs = m.get("probs_by_label", {})
        if final_label == 0:
            p = probs.get("healthy", None)
            if p is not None:
                print(f"    {name} (healthy prob): {p:.4f}")
        else:
            p = probs.get("unhealthy", None)
            if p is not None:
                print(f"    {name} (unhealthy prob): {p:.4f}")

    if "disease" in results:
        print("Disease prediction:")
        for k, v in results["disease"]["probs_by_label"].items():
            print(f"  {k}: {v:.4f}")

    # Visualize prediction probabilities
    plot_prediction_bars(binary["final_probs_by_label"], title="Final Binary Prediction")
    if "disease" in results:
        plot_prediction_bars(results["disease"]["probs_by_label"], title="Disease Prediction")


if __name__ == "__main__":
    test_image_path = "../research/data/test_images/covid6.jpeg"
    try:
        main(test_image_path)
    except ImageLoadError as exc:
        print(f"Image load error [{exc.error_code}]: {exc.message}")
