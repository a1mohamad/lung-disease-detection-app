from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from mlops.config.settings import MLOpsSettings

TFRECORDS_DIR = str(MLOpsSettings.TFRECORDS_DIR)
EXPERIMENT = MLOpsSettings.EXPERIMENT
MODEL_STAGE = MLOpsSettings.MODEL_STAGE
BATCH_SIZE = MLOpsSettings.BATCH_SIZE
EPOCHS = MLOpsSettings.EPOCHS
VAL_RATIO = MLOpsSettings.VAL_RATIO
MAX_TRAIN_BATCHES = MLOpsSettings.MAX_TRAIN_BATCHES
MAX_EVAL_BATCHES = MLOpsSettings.MAX_EVAL_BATCHES
from mlops.airflow.tasks.logging import log_model_results


with DAG(
    dag_id="log_models_monthly",
    start_date=datetime(2026, 2, 1),
    schedule="@monthly",
    catchup=False,
    tags=["mlflow", "logging"],
) as dag:
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
