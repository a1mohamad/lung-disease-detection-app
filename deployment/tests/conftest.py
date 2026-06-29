import os


# Keep test imports isolated from external infrastructure.
os.environ["DB_LOGGING_ENABLED"] = "false"
os.environ["KAFKA_ENABLED"] = "false"
os.environ["MLFLOW_ENABLED"] = "false"
os.environ["HF_MODEL_DOWNLOAD_ENABLED"] = "false"
