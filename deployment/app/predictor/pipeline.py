"""High-level lung disease detection pipeline orchestration."""

from typing import Any, Dict

from app.configs.config import AppConfig
from app.preprocessing.roi import crop_lung_roi


class LungDetection:
    """Coordinate the complete chest X-ray inference workflow.

    The pipeline owns the three model families used at serving time:
    segmentation, binary healthy/unhealthy classification, and disease subtype
    classification. It hides whether the runtime is backed by ONNX Runtime or
    Keras, so API services can call a single object without knowing how each
    model artifact is loaded.

    A prediction always produces a lung mask and binary screening result.
    Disease classification is intentionally conditional and only runs when the
    binary ensemble marks the scan as unhealthy.
    """

    def __init__(self) -> None:
        """Load all model wrappers for the configured runtime backend.

        The runtime is selected through ``AppConfig.MODEL_RUNTIME``. ONNX is the
        production path because it has a smaller serving footprint, while the
        Keras path keeps local experimentation and compatibility workflows
        available.

        Raises:
            ArtifactError or ModelError from lower-level wrappers when metadata
            or model files cannot be loaded.
        """
        if AppConfig.MODEL_RUNTIME == "onnx":
            from app.predictor.classification_onnx import (
                BinaryClassificationOnnxModel,
                EnsembleBinaryOnnxClassifier,
            )
            from app.predictor.diseases_onnx import DiseasesOnnxClassifier
            from app.predictor.segmentation_onnx import SegmentationOnnxModel

            self.seg_model = SegmentationOnnxModel(AppConfig.UNET_PATH)
            self.binary_models = {
                "densenet": BinaryClassificationOnnxModel(
                    AppConfig.DENSENET_PATH, AppConfig.MLFLOW_MODEL_NAME_DENSENET_BINARY
                ),
                "efficientnet": BinaryClassificationOnnxModel(
                    AppConfig.EFFICIENTNET_PATH, AppConfig.MLFLOW_MODEL_NAME_EFFICIENTNET_BINARY
                ),
                "inception_v3": BinaryClassificationOnnxModel(
                    AppConfig.INCEPTION_PATH, AppConfig.MLFLOW_MODEL_NAME_INCEPTION_BINARY
                ),
                "mobilenet_v3": BinaryClassificationOnnxModel(
                    AppConfig.MOBILENET_PATH, AppConfig.MLFLOW_MODEL_NAME_MOBILENET_BINARY
                ),
            }
            self.ensemble = EnsembleBinaryOnnxClassifier(self.binary_models)
            self.disease_model = DiseasesOnnxClassifier(AppConfig.DISEASE_DENSENET_PATH)
            return

        from app.predictor.classification import (
            BinaryClassificationModel,
            EnsembleBinaryClassifier,
        )
        from app.predictor.diseases import DiseasesClassifier
        from app.predictor.segmentation import SegmentationModel

        self.seg_model = SegmentationModel(AppConfig.UNET_PATH)

        self.binary_models = {
            "densenet": BinaryClassificationModel(
                AppConfig.DENSENET_PATH, AppConfig.MLFLOW_MODEL_NAME_DENSENET_BINARY
            ),
            "efficientnet": BinaryClassificationModel(
                AppConfig.EFFICIENTNET_PATH, AppConfig.MLFLOW_MODEL_NAME_EFFICIENTNET_BINARY
            ),
            "inception_v3": BinaryClassificationModel(
                AppConfig.INCEPTION_PATH, AppConfig.MLFLOW_MODEL_NAME_INCEPTION_BINARY
            ),
            "mobilenet_v3": BinaryClassificationModel(
                AppConfig.MOBILENET_PATH, AppConfig.MLFLOW_MODEL_NAME_MOBILENET_BINARY
            ),
        }

        self.ensemble = EnsembleBinaryClassifier(self.binary_models)

        self.disease_model = DiseasesClassifier(AppConfig.DISEASE_DENSENET_PATH)

    def predict(self, img, return_all: bool = True) -> Dict[str, Any]:
        """Run segmentation, binary screening, and optional disease typing.

        Args:
            img: Preprocessed image batch with shape compatible with the
                configured model metadata.
            return_all: When true, include per-model binary ensemble details in
                the response payload.

        Returns:
            A dictionary containing the predicted mask, cropped lung ROI, binary
            ensemble result, and disease subtype result for unhealthy scans.
        """

        mask = self.seg_model.predict_mask(img)

        roi_img = crop_lung_roi(img, mask, target_size=AppConfig.IMAGE_SIZE)

        # The binary ensemble remains the clinical gatekeeper for subtype
        # inference; healthy scans should not receive a disease label.
        binary = self.ensemble.predict(img, mask, return_all=return_all)

        result: Dict[str, Any] = {
            "mask": mask,
            "roi_img": roi_img,
            "binary": binary
        }

        if binary["final_label"] == 1:
            disease = self.disease_model.predict(roi_img)
            result["disease"] = disease

        return result

