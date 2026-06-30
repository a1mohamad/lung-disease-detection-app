"""Airflow DAG for scheduled monthly model evaluation logging."""

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from mlops.config.settings import MLOpsSettings
from mlops.airflow.tasks.logging import log_model_results

# Resolve settings at DAG-parse time so the Airflow UI shows the same defaults
# that scheduled tasks will use unless overridden by environment variables.
TFRECORDS_DIR = str(MLOpsSettings.TFRECORDS_DIR)
EXPERIMENT = MLOpsSettings.EXPERIMENT
MODEL_STAGE = MLOpsSettings.MODEL_STAGE
BATCH_SIZE = MLOpsSettings.BATCH_SIZE
EPOCHS = MLOpsSettings.EPOCHS
VAL_RATIO = MLOpsSettings.VAL_RATIO
MAX_TRAIN_BATCHES = MLOpsSettings.MAX_TRAIN_BATCHES
MAX_EVAL_BATCHES = MLOpsSettings.MAX_EVAL_BATCHES


# This DAG is intentionally evaluation-only: it refreshes MLflow metrics for
# deployed artifacts without retraining or publishing new model versions.
with DAG(
    dag_id="log_models_monthly",
    start_date=datetime(2026, 2, 1),
    schedule="@monthly",
    catchup=False,
    tags=["mlflow", "logging", "evaluation"],
) as dag:
    # Segmentation is logged separately from classifiers because it reports mask
    # overlap metrics instead of class probabilities.
    PythonOperator(
        task_id="log_unet_xception_segmentation",
        python_callable=log_model_results,
        op_kwargs={
            "model_name": "unet_xception_segmentation",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
        },
    )

    # Binary ensemble members are logged as independent MLflow runs so later
    # monitoring can compare drift and calibration per architecture.
    PythonOperator(
        task_id="log_densenet_binary",
        python_callable=log_model_results,
        op_kwargs={
            "model_name": "densenet_binary",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
        },
    )

    PythonOperator(
        task_id="log_efficientnet_binary",
        python_callable=log_model_results,
        op_kwargs={
            "model_name": "efficientnet_binary",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
        },
    )

    PythonOperator(
        task_id="log_inception_binary",
        python_callable=log_model_results,
        op_kwargs={
            "model_name": "inception_binary",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
        },
    )

    PythonOperator(
        task_id="log_mobilenet_binary",
        python_callable=log_model_results,
        op_kwargs={
            "model_name": "mobilenet_binary",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
        },
    )

    # The disease subtype model is evaluated on the filtered unhealthy subset,
    # which is handled inside the shared logging task by model specification.
    PythonOperator(
        task_id="log_densenet_diseases",
        python_callable=log_model_results,
        op_kwargs={
            "model_name": "densenet_diseases",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
        },
    )

