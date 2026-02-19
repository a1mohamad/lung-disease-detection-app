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
from mlops.airflow.tasks.retrain import retrain_model


with DAG(
    dag_id="retrain_binary_densenet",
    start_date=datetime(2026, 2, 1),
    schedule="@monthly",
    catchup=False,
    tags=["mlflow", "retrain", "binary"],
) as dag:
    PythonOperator(
        task_id="retrain_binary_densenet",
        python_callable=retrain_model,
        op_kwargs={
            "model_name": "densenet_binary",
            "tfrecords_dir": TFRECORDS_DIR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_train_batches": MAX_TRAIN_BATCHES,
            "max_eval_batches": MAX_EVAL_BATCHES,
            "experiment": EXPERIMENT,
            "stage": MODEL_STAGE,
            "val_ratio": VAL_RATIO,
        },
    )
