from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import mlflow

from mlops.config.model_specs import MODEL_SPECS, POST_HOC_SPECS
from mlops.config.settings import MLOpsSettings
from mlops.core.backfill.notebook_params import collect_notebook_params
from mlops.core.data.tfrecord_ops import list_tfrecords, split_tfrecords
from mlops.core.evaluation.runner import build_eval_dataset_for_spec, evaluate_model_for_spec
from mlops.core.tracking.mlflow_io import flatten_dict, load_compiled_model, load_yaml

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_optuna_params(paths: list[Path]) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in ("best_hparams", "phase1_settings", "phase2_settings"):
            if isinstance(payload.get(key), dict):
                params.update(flatten_dict(payload[key], f"optuna.{key}"))
        if "best_value" in payload:
            params["optuna.best_value"] = str(payload["best_value"])
        if "best_trial_number" in payload:
            params["optuna.best_trial_number"] = str(payload["best_trial_number"])
    return params


def run_for_model(
    spec,
    post_hoc_spec,
    tfrecords_dir: Path | None,
    batch_size: int,
    max_eval_batches: int | None,
    stage: str,
    experiment: str,
    val_ratio: float,
    register_model: bool,
    with_eval: bool,
) -> None:
    mlflow.set_experiment(experiment)
    metadata = load_yaml(spec.metadata_path)

    with mlflow.start_run(run_name=f"{spec.task}:{spec.name}"):
        mlflow.set_tags(
            {
                "task": spec.task,
                "model_name": spec.name,
                "run_type": "post_hoc_backfill",
                "with_eval": str(with_eval).lower(),
            }
        )
        mlflow.log_param("with_eval", str(with_eval).lower())
        if with_eval:
            mlflow.set_tag("tfrecords_dir", str(tfrecords_dir))
            mlflow.log_params(
                {
                    "batch_size": batch_size,
                    "max_eval_batches": max_eval_batches,
                    "val_ratio": val_ratio,
                }
            )

        mlflow.log_artifact(str(spec.metadata_path), artifact_path="metadata")
        for optuna_path in post_hoc_spec.optuna_jsons:
            if optuna_path.exists():
                mlflow.log_artifact(str(optuna_path), artifact_path="optuna")
        for notebook_path in post_hoc_spec.notebooks:
            if notebook_path.exists():
                mlflow.log_artifact(str(notebook_path), artifact_path="notebooks")

        notebook_params = collect_notebook_params(post_hoc_spec.notebooks)
        if notebook_params:
            mlflow.log_params(notebook_params)

        optuna_params = load_optuna_params(post_hoc_spec.optuna_jsons)
        if optuna_params:
            mlflow.log_params(optuna_params)

        if isinstance(metadata, dict):
            mlflow.log_params(flatten_dict(metadata))
            reported_metrics = metadata.get("metrics", {})
            if isinstance(reported_metrics, dict):
                for key, value in reported_metrics.items():
                    try:
                        mlflow.log_metric(f"reported_{key}", float(value))
                    except Exception:
                        continue

        model, _ = load_compiled_model(spec, stage, metadata)

        if with_eval:
            if tfrecords_dir is None:
                raise RuntimeError("with_eval=true requires --tfrecords-dir.")
            all_files = list_tfrecords(tfrecords_dir)
            _, val_files = split_tfrecords(all_files, val_ratio)
            if not val_files:
                raise RuntimeError("No TFRecord files found for validation.")
            val_ds = build_eval_dataset_for_spec(spec, metadata, val_files, batch_size)
            metrics = evaluate_model_for_spec(spec, model, val_ds, max_eval_batches, metadata)

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

        if register_model:
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                registered_model_name=spec.registered_name,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time post-hoc backfill to MLflow.")
    parser.add_argument("--tfrecords-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=MLOpsSettings.BATCH_SIZE)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--experiment", type=str, default=MLOpsSettings.EXPERIMENT)
    parser.add_argument("--stage", type=str, default=MLOpsSettings.MODEL_STAGE)
    parser.add_argument("--val-ratio", type=float, default=MLOpsSettings.VAL_RATIO)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--with-eval", action="store_true")
    parser.add_argument("--model-name", type=str, default="")
    return parser.parse_args()


def run_pipeline(
    *,
    tfrecords_dir: str | None,
    batch_size: int,
    max_eval_batches: int | None,
    experiment: str,
    stage: str,
    val_ratio: float,
    register_model: bool,
    with_eval: bool,
    model_name: str | None = None,
) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    tfrecords_dir_path = Path(tfrecords_dir) if tfrecords_dir else None

    model_by_name = {spec.name: spec for spec in MODEL_SPECS}
    post_hoc_by_name = {spec.model_name: spec for spec in POST_HOC_SPECS}

    target_names = [model_name] if model_name else list(post_hoc_by_name.keys())
    for name in target_names:
        if not name:
            continue
        if name not in model_by_name or name not in post_hoc_by_name:
            raise ValueError(f"Unknown model name for post-hoc backfill: {name}")
        run_for_model(
            spec=model_by_name[name],
            post_hoc_spec=post_hoc_by_name[name],
            tfrecords_dir=tfrecords_dir_path,
            batch_size=batch_size,
            max_eval_batches=max_eval_batches,
            stage=stage,
            experiment=experiment,
            val_ratio=val_ratio,
            register_model=register_model,
            with_eval=with_eval,
        )


def main() -> None:
    args = parse_args()
    max_eval_batches = args.max_eval_batches or None
    tfrecords_dir = args.tfrecords_dir or str(MLOpsSettings.TFRECORDS_DIR)
    run_pipeline(
        tfrecords_dir=tfrecords_dir if args.with_eval else None,
        batch_size=args.batch_size,
        max_eval_batches=max_eval_batches,
        experiment=args.experiment,
        stage=args.stage,
        val_ratio=args.val_ratio,
        register_model=args.register_model,
        with_eval=args.with_eval,
        model_name=args.model_name or None,
    )


if __name__ == "__main__":
    main()

