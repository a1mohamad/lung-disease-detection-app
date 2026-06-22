from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

from mlops.airflow.tasks.prepare_data import prepare_retrain_dataset


DATASET_CONF = {
    "tfrecords_dir": "{{ ti.xcom_pull(task_ids='prepare_dataset')['snapshot_dir'] }}",
    "dataset_mode": "{{ ti.xcom_pull(task_ids='prepare_dataset')['dataset_mode'] }}",
}


with DAG(
    dag_id="orchestrate_retrain_pipeline",
    start_date=datetime(2026, 2, 1),
    schedule="@monthly",
    catchup=False,
    tags=["mlflow", "orchestration"],
) as dag:
    start = EmptyOperator(task_id="start")
    prepare_dataset = PythonOperator(
        task_id="prepare_dataset",
        python_callable=prepare_retrain_dataset,
    )

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

    start >> prepare_dataset >> trigger_segmentation
    trigger_segmentation >> [
        trigger_binary_densenet,
        trigger_binary_efficientnet,
        trigger_binary_inception,
        trigger_binary_mobilenet,
    ] >> binary_done
    binary_done >> trigger_diseases >> end

