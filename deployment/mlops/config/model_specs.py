"""Model registry metadata for evaluation, retraining, and release jobs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List

# PROJECT_ROOT can point either at the deployment folder or the repository root;
# this detection keeps Airflow containers and local scripts using one config.
ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
DEPLOYMENT_DIR = ROOT
if not (DEPLOYMENT_DIR / "saved_models").exists() and (ROOT / "deployment" / "saved_models").exists():
    DEPLOYMENT_DIR = ROOT / "deployment"

# Research paths are used only for historical backfill artifacts, not notebook
# execution, so missing research folders do not affect runtime inference.
RESEARCH_DIR = ROOT / "research"
if not RESEARCH_DIR.exists() and (ROOT.parent / "research").exists():
    RESEARCH_DIR = ROOT.parent / "research"

SAVED_MODELS_DIR = DEPLOYMENT_DIR / "saved_models"


@dataclass(frozen=True)
class ModelSpec:
    """Deployment and registry metadata for one managed model.

    Attributes:
        name: Internal short name used by CLI arguments and DAGs.
        task: Training/evaluation task family.
        model_dir: Directory containing the deployable model artifact.
        metadata_path: Path to the model metadata contract.
        registered_name: MLflow registered model name.
        promotion_metric: Metric key used when comparing retrained candidates.
    """

    name: str
    task: str
    model_dir: Path
    metadata_path: Path
    registered_name: str
    promotion_metric: str


@dataclass(frozen=True)
class PostHocSpec:
    """Research artifacts used for one-time MLflow backfill runs.

    Attributes:
        model_name: Internal model short name.
        notebooks: Research notebooks logged as historical context.
        optuna_jsons: Optional Optuna result JSON files logged with the run.
    """

    model_name: str
    notebooks: List[Path]
    optuna_jsons: List[Path]


MODEL_SPECS = [
    # Binary screening models are managed independently because the runtime
    # ensemble needs per-architecture metrics and release artifacts.
    ModelSpec(
        name="densenet_binary",
        task="binary_classification",
        model_dir=SAVED_MODELS_DIR / "healthy_unhealthy" / "densenet",
        metadata_path=SAVED_MODELS_DIR / "healthy_unhealthy" / "densenet" / "metadata.yaml",
        registered_name="lung-binary-densenet",
        promotion_metric="val_f1",
    ),
    ModelSpec(
        name="efficientnet_binary",
        task="binary_classification",
        model_dir=SAVED_MODELS_DIR / "healthy_unhealthy" / "efficientnet",
        metadata_path=SAVED_MODELS_DIR / "healthy_unhealthy" / "efficientnet" / "metadata.yaml",
        registered_name="lung-binary-efficientnet",
        promotion_metric="val_f1",
    ),
    ModelSpec(
        name="inception_binary",
        task="binary_classification",
        model_dir=SAVED_MODELS_DIR / "healthy_unhealthy" / "inception",
        metadata_path=SAVED_MODELS_DIR / "healthy_unhealthy" / "inception" / "metadata.yaml",
        registered_name="lung-binary-inception",
        promotion_metric="val_f1",
    ),
    ModelSpec(
        name="mobilenet_binary",
        task="binary_classification",
        model_dir=SAVED_MODELS_DIR / "healthy_unhealthy" / "mobilenet",
        metadata_path=SAVED_MODELS_DIR / "healthy_unhealthy" / "mobilenet" / "metadata.yaml",
        registered_name="lung-binary-mobilenet",
        promotion_metric="val_f1",
    ),
    ModelSpec(
        # The disease model is only trained/evaluated on abnormal scans and is
        # promoted with multiclass F1.
        name="densenet_diseases",
        task="multiclass_classification",
        model_dir=SAVED_MODELS_DIR / "diseases" / "densenet",
        metadata_path=SAVED_MODELS_DIR / "diseases" / "densenet" / "metadata.yaml",
        registered_name="lung-diseases-densenet",
        promotion_metric="val_f1",
    ),
    ModelSpec(
        # Segmentation promotion uses Dice because mask overlap, not class F1,
        # is the clinically relevant quality signal.
        name="unet_xception_segmentation",
        task="segmentation",
        model_dir=SAVED_MODELS_DIR / "segmentation" / "unet_xception",
        metadata_path=SAVED_MODELS_DIR / "segmentation" / "unet_xception" / "metadata.yaml",
        registered_name="lung-segmentation-unet-xception",
        promotion_metric="dice_coefficient",
    ),
]


POST_HOC_SPECS = [
    # Post-hoc specs preserve the original research notebooks and Optuna JSONs
    # as MLflow context for models trained before the automated pipeline.
    PostHocSpec(
        model_name="densenet_binary",
        notebooks=[
            RESEARCH_DIR / "binary-healthy_unhealthy" / "densenet" / "DenseNet121-healthy_binary_classification.ipynb",
            RESEARCH_DIR / "binary-healthy_unhealthy" / "densenet" / "optuna" / "DenseNet121-healthy_unhealthy-phase1_architecture.ipynb",
            RESEARCH_DIR / "binary-healthy_unhealthy" / "densenet" / "optuna" / "DenseNet121-healthy_unhealthy-phase2-optimization_params.ipynb",
        ],
        optuna_jsons=[
            RESEARCH_DIR / "binary-healthy_unhealthy" / "densenet" / "optuna" / "phase1_architecture" / "healthy_unhealthy-best_architecture.json",
            RESEARCH_DIR / "binary-healthy_unhealthy" / "densenet" / "optuna" / "phase2_optimization" / "healthy_unhealthy-best_hparams.json",
        ],
    ),
    PostHocSpec(
        model_name="efficientnet_binary",
        notebooks=[
            RESEARCH_DIR / "binary-healthy_unhealthy" / "EfficientNetV2B3-healthy-binary_classification.ipynb",
        ],
        optuna_jsons=[],
    ),
    PostHocSpec(
        model_name="inception_binary",
        notebooks=[
            RESEARCH_DIR / "binary-healthy_unhealthy" / "InceptionV3-healthy-binary_classification.ipynb",
        ],
        optuna_jsons=[],
    ),
    PostHocSpec(
        model_name="mobilenet_binary",
        notebooks=[
            RESEARCH_DIR / "binary-healthy_unhealthy" / "MobileNetV3-healthy-binary_classification.ipynb",
        ],
        optuna_jsons=[],
    ),
    PostHocSpec(
        model_name="densenet_diseases",
        notebooks=[
            RESEARCH_DIR / "multiclass-diseases" / "DenseNet121-diseases-multiclass_classification.ipynb",
            RESEARCH_DIR / "multiclass-diseases" / "optuna" / "DenseNet121-diseases-phase1-architecture.ipynb",
            RESEARCH_DIR / "multiclass-diseases" / "optuna" / "DenseNet121-diseases-phase2-optimization_params.ipynb",
        ],
        optuna_jsons=[
            RESEARCH_DIR / "multiclass-diseases" / "optuna" / "phase1_architecture" / "diseases-best_architecture.json",
            RESEARCH_DIR / "multiclass-diseases" / "optuna" / "phase2_optimization" / "diseases-best_hparams.json",
        ],
    ),
    PostHocSpec(
        model_name="unet_xception_segmentation",
        notebooks=[
            RESEARCH_DIR / "segmentation" / "segemntation-U-Net_Xception.ipynb",
            RESEARCH_DIR / "segmentation" / "segmentation-checking_multitask_model.ipynb",
        ],
        optuna_jsons=[],
    ),
]
