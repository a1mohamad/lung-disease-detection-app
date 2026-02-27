from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from mlops.config.model_specs import MODEL_SPECS
from mlops.config.settings import MLOpsSettings
from mlops.core.data.tfrecord_ops import list_tfrecords, split_tfrecords
from mlops.core.models.loader import load_compiled_model
from mlops.core.tracking.mlflow_io import flatten_dict, load_yaml
from mlops.core.evaluation.runner import build_eval_dataset_for_spec, evaluate_model_for_spec
from mlops.core.tracking.evaluation_dataset import get_or_create_eval_dataset
from mlops.core.tracking.run_summary import build_run_summary, log_run_summary

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def log_results_for_model(
    spec,
    tfrecords_dir: Path,
    batch_size: int,
    max_eval_batches: int | None,
    stage: str,
    experiment: str,
    val_ratio: float,
) -> None:
    mlflow.set_experiment(experiment)
    metadata = load_yaml(spec.metadata_path)
    client = MlflowClient()

    with mlflow.start_run(run_name=f"{spec.task}:{spec.name}"):
        run = mlflow.active_run()
        run_params = {
            "batch_size": batch_size,
            "max_eval_batches": max_eval_batches,
            "val_ratio": val_ratio,
        }
        experiment_id = run.info.experiment_id if run else None
        dataset_id = (
            get_or_create_eval_dataset(
                client=client,
                experiment_id=experiment_id,
                spec_name=spec.name,
                stage=stage,
                val_ratio=val_ratio,
                tfrecords_dir=tfrecords_dir,
            )
            if experiment_id
            else None
        )
        mlflow.set_tags(
            {
                "task": spec.task,
                "model_name": spec.name,
                "run_type": "monthly_evaluation",
                "run_mode": "evaluation",
                "mlflow.run.kind": "evaluation",
                "tfrecords_dir": str(tfrecords_dir),
            }
        )
        if dataset_id:
            mlflow.set_tag("evaluation_dataset_id", dataset_id)

        mlflow.log_params(run_params)

        mlflow.log_artifact(str(spec.metadata_path), artifact_path="metadata")

        reported_metrics: dict[str, object] = {}
        if isinstance(metadata, dict):
            mlflow.log_params(flatten_dict(metadata))
            raw_reported_metrics = metadata.get("metrics", {})
            if isinstance(raw_reported_metrics, dict):
                reported_metrics = raw_reported_metrics
                for key, value in raw_reported_metrics.items():
                    try:
                        mlflow.log_metric(f"reported_{key}", float(value))
                    except Exception:
                        continue

        model, _ = load_compiled_model(spec, stage, metadata)

        all_files = list_tfrecords(tfrecords_dir)
        _, val_files = split_tfrecords(all_files, val_ratio)
        if not val_files:
            raise RuntimeError("No TFRecord files found for validation.")

        val_ds = build_eval_dataset_for_spec(spec, metadata, val_files, batch_size)
        metrics = evaluate_model_for_spec(spec, model, val_ds, max_eval_batches, metadata)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        summary = build_run_summary(
            run_type="monthly_evaluation",
            model_name=spec.name,
            task=spec.task,
            metrics=metrics,
            params=run_params,
            reported_metrics=reported_metrics,
            extra={
                "run_id": run.info.run_id if run else None,
                "stage": stage,
                "dataset_id": dataset_id,
            },
        )
        log_run_summary(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly logging of model results.")
    parser.add_argument("--tfrecords-dir", type=str, default=str(MLOpsSettings.TFRECORDS_DIR))
    parser.add_argument("--batch-size", type=int, default=MLOpsSettings.BATCH_SIZE)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--experiment", type=str, default=MLOpsSettings.EXPERIMENT)
    parser.add_argument("--stage", type=str, default=MLOpsSettings.MODEL_STAGE)
    parser.add_argument("--val-ratio", type=float, default=MLOpsSettings.VAL_RATIO)
    return parser.parse_args()


def run_pipeline(
    *,
    tfrecords_dir: str,
    batch_size: int,
    max_eval_batches: int | None,
    experiment: str,
    stage: str,
    model_name: str | None = None,
    val_ratio: float = 0.2,
) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    tfrecords_dir_path = Path(tfrecords_dir)

    for spec in MODEL_SPECS:
        if model_name and spec.name != model_name:
            continue
        log_results_for_model(
            spec=spec,
            tfrecords_dir=tfrecords_dir_path,
            batch_size=batch_size,
            max_eval_batches=max_eval_batches,
            stage=stage,
            experiment=experiment,
            val_ratio=val_ratio,
        )


def main() -> None:
    args = parse_args()
    max_eval_batches = args.max_eval_batches or None
    run_pipeline(
        tfrecords_dir=args.tfrecords_dir,
        batch_size=args.batch_size,
        max_eval_batches=max_eval_batches,
        experiment=args.experiment,
        stage=args.stage,
        model_name=None,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
