"""Airflow DAG for retraining the binary InceptionV3 model."""

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from mlops.config.settings import MLOpsSettings

# Constants are read from settings once during DAG parsing so the Airflow UI
# exposes the effective defaults for each retraining task.
TFRECORDS_DIR = str(MLOpsSettings.TFRECORDS_DIR)
EXPERIMENT = MLOpsSettings.EXPERIMENT
MODEL_STAGE = MLOpsSettings.MODEL_STAGE
BATCH_SIZE = MLOpsSettings.BATCH_SIZE
EPOCHS = MLOpsSettings.EPOCHS
VAL_RATIO = MLOpsSettings.VAL_RATIO
MAX_TRAIN_BATCHES = MLOpsSettings.MAX_TRAIN_BATCHES
MAX_EVAL_BATCHES = MLOpsSettings.MAX_EVAL_BATCHES
DATASET_MODE = MLOpsSettings.RETRAIN_DATASET_MODE
# Import after settings so local DAG parsing can fail early on configuration
# issues before loading heavier TensorFlow task code.
from mlops.airflow.tasks.retrain import retrain_model


with DAG(
    dag_id="retrain_binary_inception",
    start_date=datetime(2026, 2, 1),
    schedule=None,
    catchup=False,
    tags=["mlflow", "retrain", "binary"],
) as dag:
    # The orchestrator passes reviewed snapshot overrides through dag_run.conf;
    # the task helper merges those overrides with the defaults below.
    PythonOperator(
        task_id="retrain_binary_inception",
        python_callable=retrain_model,
        op_kwargs={
            "model_name": "inception_binary",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_train_batches": MAX_TRAIN_BATCHES,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
            "dataset_mode": DATASET_MODE,
        },
    )


