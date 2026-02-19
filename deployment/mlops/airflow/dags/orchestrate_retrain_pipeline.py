from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


with DAG(
    dag_id="orchestrate_retrain_pipeline",
    start_date=datetime(2026, 2, 1),
    schedule="@monthly",
    catchup=False,
    tags=["mlflow", "orchestration"],
) as dag:
    start = EmptyOperator(task_id="start")

    trigger_segmentation = TriggerDagRunOperator(
        task_id="trigger_segmentation",
        trigger_dag_id="retrain_unet_xception_segmentation",
        wait_for_completion=True,
        reset_dag_run=True,
    )

    trigger_binary_densenet = TriggerDagRunOperator(
        task_id="trigger_binary_densenet",
        trigger_dag_id="retrain_binary_densenet",
        wait_for_completion=True,
        reset_dag_run=True,
    )
    trigger_binary_efficientnet = TriggerDagRunOperator(
        task_id="trigger_binary_efficientnet",
        trigger_dag_id="retrain_binary_efficientnet",
        wait_for_completion=True,
        reset_dag_run=True,
    )
    trigger_binary_inception = TriggerDagRunOperator(
        task_id="trigger_binary_inception",
        trigger_dag_id="retrain_binary_inception",
        wait_for_completion=True,
        reset_dag_run=True,
    )
    trigger_binary_mobilenet = TriggerDagRunOperator(
        task_id="trigger_binary_mobilenet",
        trigger_dag_id="retrain_binary_mobilenet",
        wait_for_completion=True,
        reset_dag_run=True,
    )

    binary_done = EmptyOperator(task_id="binary_done")

    trigger_diseases = TriggerDagRunOperator(
        task_id="trigger_diseases",
        trigger_dag_id="retrain_diseases_densenet",
        wait_for_completion=True,
        reset_dag_run=True,
    )

    end = EmptyOperator(task_id="end")

    start >> trigger_segmentation
    trigger_segmentation >> [
        trigger_binary_densenet,
        trigger_binary_efficientnet,
        trigger_binary_inception,
        trigger_binary_mobilenet,
    ] >> binary_done
    binary_done >> trigger_diseases >> end
