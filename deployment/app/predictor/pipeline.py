from typing import Any, Dict
import tensorflow as tf

from app.configs.config import AppConfig
from app.preprocessing.roi import crop_lung_roi

from app.predictor.segmentation import SegmentationModel
from app.predictor.classification import (BinaryClassificationModel, 
                                          EnsembleBinaryClassifier)
from app.predictor.diseases import DiseasesClassifier


class LungDetection:
    def __init__(self) -> None:
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

    def predict(self, img: tf.Tensor, return_all: bool = True) -> Dict[str, Any]:

        mask = self.seg_model.predict_mask(img)

        roi_img = crop_lung_roi(img, mask, target_size=AppConfig.IMAGE_SIZE)

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

