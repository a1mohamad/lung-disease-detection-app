from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mlops.config.model_specs import MODEL_SPECS
from mlops.config.settings import MLOpsSettings
from mlops.core.data.tfrecord_ops import list_tfrecords, split_tfrecords
from mlops.core.models.loader import load_compiled_model
from mlops.core.tracking.mlflow_io import flatten_dict, load_yaml
from mlops.core.tracking.registry import promote_if_better
from mlops.core.training.retrain import retrain_and_evaluate_for_spec

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logger = logging.getLogger(__name__)


def run_for_model(
    spec,
    tfrecords_dir: Path,
    batch_size: int,
    epochs: int,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    stage: str,
    experiment: str,
    val_ratio: float,
    register_model: bool,
) -> None:
    metadata = load_yaml(spec.metadata_path)
    model, _ = load_compiled_model(spec, stage, metadata)

    all_files = list_tfrecords(tfrecords_dir)
    train_files, val_files = split_tfrecords(all_files, val_ratio)
    if not train_files or not val_files:
        raise RuntimeError("No TFRecord files found for training/validation.")

    model, metrics = retrain_and_evaluate_for_spec(
        spec=spec,
        metadata=metadata,
        model=model,
        train_files=train_files,
        val_files=val_files,
        batch_size=batch_size,
        epochs=epochs,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
    )

    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"{spec.task}:{spec.name}") as run:
            mlflow.set_tags(
                {
                    "task": spec.task,
                    "model_name": spec.name,
                    "run_type": "monthly_retrain",
                    "tfrecords_dir": str(tfrecords_dir),
                }
            )

            mlflow.log_params(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "max_train_batches": max_train_batches,
                    "max_eval_batches": max_eval_batches,
                    "val_ratio": val_ratio,
                }
            )

            mlflow.log_artifact(str(spec.metadata_path), artifact_path="metadata")

            if isinstance(metadata, dict):
                mlflow.log_params(flatten_dict(metadata))
                reported_metrics = metadata.get("metrics", {})
                if isinstance(reported_metrics, dict):
                    for key, value in reported_metrics.items():
                        try:
                            mlflow.log_metric(f"reported_{key}", float(value))
                        except Exception:
                            continue

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.keras.log_model(
                model,
                name="model",
                registered_model_name=spec.registered_name if register_model else None,
            )
            mlflow.log_param("register_model", register_model)

            if register_model:
                promote_if_better(spec.registered_name, run.info.run_id, spec.promotion_metric)
    except Exception as exc:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fallback_dir = spec.model_dir / "retrained_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / f"{spec.name}-{timestamp}.keras"
        model.save(str(fallback_path))
        logger.warning(
            "MLflow logging/registration failed; saved retrained model locally. model=%s path=%s error=%s",
            spec.name,
            fallback_path,
            exc,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly retraining + MLflow logging.")
    parser.add_argument("--tfrecords-dir", type=str, default=str(MLOpsSettings.TFRECORDS_DIR))
    parser.add_argument("--batch-size", type=int, default=MLOpsSettings.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=MLOpsSettings.EPOCHS)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--experiment", type=str, default=MLOpsSettings.EXPERIMENT)
    parser.add_argument("--stage", type=str, default=MLOpsSettings.MODEL_STAGE)
    parser.add_argument("--val-ratio", type=float, default=MLOpsSettings.VAL_RATIO)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--allow-manual-run", action="store_true")
    return parser.parse_args()


def run_pipeline(
    *,
    tfrecords_dir: str,
    batch_size: int,
    epochs: int,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    experiment: str,
    stage: str,
    model_name: str,
    val_ratio: float = 0.2,
    register_model: bool = False,
) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    tfrecords_dir_path = Path(tfrecords_dir)

    matched = False
    for spec in MODEL_SPECS:
        if spec.name != model_name:
            continue
        matched = True
        run_for_model(
            spec=spec,
            tfrecords_dir=tfrecords_dir_path,
            batch_size=batch_size,
            epochs=epochs,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            stage=stage,
            experiment=experiment,
            val_ratio=val_ratio,
            register_model=register_model,
        )
    if not matched:
        raise ValueError(f"Unknown model_name: {model_name}")


def main() -> None:
    args = parse_args()
    if not os.getenv("AIRFLOW_CTX_DAG_ID") and not args.allow_manual_run:
        raise RuntimeError(
            "Manual monthly_retrain execution is blocked by default. "
            "Use Airflow trigger, or pass --allow-manual-run intentionally."
        )
    max_train_batches = args.max_train_batches or None
    max_eval_batches = args.max_eval_batches or None
    run_pipeline(
        tfrecords_dir=args.tfrecords_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        experiment=args.experiment,
        stage=args.stage,
        model_name=args.model_name,
        val_ratio=args.val_ratio,
        register_model=args.register_model,
    )


if __name__ == "__main__":
    main()
