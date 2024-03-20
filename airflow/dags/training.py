from __future__ import annotations

import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import pythonOperator
from src.ocr_captcha.pipeline.training_pipeline import TrainingPipeline

t_pipe = TrainingPipeline(False, "model.weights.h5")

with DAG(
    "ocr_captcha_training",
    default_args={"retries": 2},
    description="This is the training pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2024, 3, 17, tz="UTC"),
    catchup=False,
    tags=["machineLearning", "ImageClasification", "ocr_Captcha"],
) as dag:
    
    dag.doc_md = __doc__
    
    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        train_data_path_x, train_data_path_y, test_data_path_x, test_data_path_y, unique_chars = t_pipe.start_data_ingestion()
        ti.xcom_push("data_ingestion_artifacts", {
            "train_data_path_x": train_data_path_x,
            "train_data_path_y": train_data_path_y,
            "test_data_path_x": test_data_path_x,
            "test_data_path_y": test_data_path_y,
            "unique_chars": unique_chars
        })
        
    def model_trainer(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_artifacts = ti.xcom_pull(
            task_ids="data_ingestion",
            key="data_ingestion_artifacts"
            )
        train_path_x = data_ingestion_artifacts["train_data_path_x"]
        train_path_y = data_ingestion_artifacts["train_data_path_y"]
        test_path_x = data_ingestion_artifacts["test_data_path_x"]
        test_path_y = data_ingestion_artifacts["test_data_path_y"]
        unique_chars = data_ingestion_artifacts["uniqe_chars"]
        t_pipe.initiate_training()
        t_pipe.train(train_path_x, train_path_y, test_path_x, test_path_y, unique_chars)
     
     
    def push_data_to_azure(**kwargs):
        print("To be implemented")
    
    # Task Instances
    data_ingestion_task = pythonOperator(
        task_id = "data_ingestion",
        python_callable = data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """
        ### Ingestion Task
        This task ingests the data and creates the train-test x and y files
        """
    )
    
    model_trainer_task = pythonOperator(
        task_id = "model_trainer",
        python_callable = model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """
        ### Model Trainer Task
        This task trains a machine learning model
        """
    )
    push_to_azure_task = pythonOperator(
        task_id="push_to_azure",
        python_callable=push_data_to_azure,
    )

data_ingestion_task >> model_trainer_task >> push_to_azure_task

