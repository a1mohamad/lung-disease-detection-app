"""Airflow DAG that prepares data and orchestrates all retraining DAGs."""

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

from mlops.airflow.tasks.prepare_data import prepare_retrain_dataset


# Downstream DAGs receive the prepared snapshot through dag_run.conf. Jinja pulls
# from XCom at runtime so every model retrains on the same reviewed-data cut.
DATASET_CONF = {
    "tfrecords_dir": "{{ ti.xcom_pull(task_ids='prepare_dataset')['snapshot_dir'] }}",
    "dataset_mode": "{{ ti.xcom_pull(task_ids='prepare_dataset')['dataset_mode'] }}",
}


# The orchestrator owns scheduling and dependency order; model-specific DAGs
# stay unscheduled so they can also be triggered manually for focused retrains.
with DAG(
    dag_id="orchestrate_retrain_pipeline",
    start_date=datetime(2026, 2, 1),
    schedule="@monthly",
    catchup=False,
    tags=["mlflow", "orchestration"],
) as dag:
    start = EmptyOperator(task_id="start")
    # Dataset preparation snapshots reviewed records before any training starts,
    # preventing models in the same monthly cycle from seeing different data.
    prepare_dataset = PythonOperator(
        task_id="prepare_dataset",
        python_callable=prepare_retrain_dataset,
    )

    # Segmentation runs first because ROI and mask quality are foundational for
    # the classifier preprocessing and reviewer-facing artifacts.
    trigger_segmentation = TriggerDagRunOperator(
        task_id="trigger_segmentation",
        trigger_dag_id="retrain_unet_xception_segmentation",
        trigger_run_id="orchestrate__{{ dag_run.run_id }}__retrain_unet_xception_segmentation",
        wait_for_completion=True,
        reset_dag_run=False,
        allowed_states=["success"],
        failed_states=["failed"],
        poke_interval=15,
        conf=DATASET_CONF,
    )

    # Binary classifiers fan out in parallel after segmentation succeeds. Each
    # triggered DAG waits for completion so the orchestrator can fail fast.
    trigger_binary_densenet = TriggerDagRunOperator(
        task_id="trigger_binary_densenet",
        trigger_dag_id="retrain_binary_densenet",
        trigger_run_id="orchestrate__{{ dag_run.run_id }}__retrain_binary_densenet",
        wait_for_completion=True,
        reset_dag_run=False,
        allowed_states=["success"],
        failed_states=["failed"],
        poke_interval=15,
        conf=DATASET_CONF,
    )
    trigger_binary_efficientnet = TriggerDagRunOperator(
        task_id="trigger_binary_efficientnet",
        trigger_dag_id="retrain_binary_efficientnet",
        trigger_run_id="orchestrate__{{ dag_run.run_id }}__retrain_binary_efficientnet",
        wait_for_completion=True,
        reset_dag_run=False,
        allowed_states=["success"],
        failed_states=["failed"],
        poke_interval=15,
        conf=DATASET_CONF,
    )
    trigger_binary_inception = TriggerDagRunOperator(
        task_id="trigger_binary_inception",
        trigger_dag_id="retrain_binary_inception",
        trigger_run_id="orchestrate__{{ dag_run.run_id }}__retrain_binary_inception",
        wait_for_completion=True,
        reset_dag_run=False,
        allowed_states=["success"],
        failed_states=["failed"],
        poke_interval=15,
        conf=DATASET_CONF,
    )
    trigger_binary_mobilenet = TriggerDagRunOperator(
        task_id="trigger_binary_mobilenet",
        trigger_dag_id="retrain_binary_mobilenet",
        trigger_run_id="orchestrate__{{ dag_run.run_id }}__retrain_binary_mobilenet",
        wait_for_completion=True,
        reset_dag_run=False,
        allowed_states=["success"],
        failed_states=["failed"],
        poke_interval=15,
        conf=DATASET_CONF,
    )

    binary_done = EmptyOperator(task_id="binary_done")

    # Disease subtype retraining waits for binary retraining to finish because
    # production serving only calls it after an unhealthy binary decision.
    trigger_diseases = TriggerDagRunOperator(
        task_id="trigger_diseases",
        trigger_dag_id="retrain_diseases_densenet",
        trigger_run_id="orchestrate__{{ dag_run.run_id }}__retrain_diseases_densenet",
        wait_for_completion=True,
        reset_dag_run=False,
        allowed_states=["success"],
        failed_states=["failed"],
        poke_interval=15,
        conf=DATASET_CONF,
    )

    end = EmptyOperator(task_id="end")

    # The dependency graph documents the clinical flow: prepare data, validate
    # segmentation, refresh binary screening models, then refresh subtyping.
    start >> prepare_dataset >> trigger_segmentation
    trigger_segmentation >> [
        trigger_binary_densenet,
        trigger_binary_efficientnet,
        trigger_binary_inception,
        trigger_binary_mobilenet,
    ] >> binary_done
    binary_done >> trigger_diseases >> end

