'''
Centralized Configuration Module.
Handles path management for the app, assets, and saved_models.
Uses absolute paths derived from the project root for stability.
'''

import os
from urllib.parse import quote_plus
from pathlib import Path


class AppConfig:
    # 1. BASE PATHS
    # This points to the folder containing 'app/', 'assets/', and 'saved_models/'
    # Since this file is in app/configs/config.py, we go up 3 levels to project root.
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent

    # 2. TOP-LEVEL DIRECTORIES
    APP_DIR = ROOT_DIR / "app"
    ASSETS_DIR = ROOT_DIR / "assets"
    MODELS_ROOT = ROOT_DIR / "saved_models"

    # 3. SPECIFIC MODEL CATEGORIES
    CLASSIFICATION_DIR = MODELS_ROOT / "healthy_unhealthy"
    DISEASES_DIR = MODELS_ROOT / "diseases"
    SEGMENTATION_DIR = MODELS_ROOT / "segmentation"

    # 4. MODEL SPECIFIC PATHS (Deeply Nested)
    # Healthy vs Unhealthy
    DENSENET_PATH = CLASSIFICATION_DIR / "densenet"
    EFFICIENTNET_PATH = CLASSIFICATION_DIR / "efficientnet"
    INCEPTION_PATH = CLASSIFICATION_DIR / "inception"
    MOBILENET_PATH = CLASSIFICATION_DIR / "mobilenet"


    # Diseases
    DISEASE_DENSENET_PATH = DISEASES_DIR / "densenet"

    # Segmentation
    UNET_PATH = SEGMENTATION_DIR / "unet_xception"

    # 5. ASSETS & MAPPINGS
    CLASSIFICATION_JSON = ASSETS_DIR / "healthy_unhealthy_mapping.json"
    DISEASES_JSON = ASSETS_DIR / "diseases_mapping.json"
    PREDICTION_DIR = ASSETS_DIR / "predictions"

    # 6. GLOBAL PARAMETERS
    IMAGE_SIZE = (256, 256)
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "1433")
    DB_NAME = os.getenv("DB_NAME", "lung_detection")
    DB_USER = os.getenv("DB_USER", "sa")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
    DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")
    DB_TRUST_SERVER_CERTIFICATE = os.getenv("DB_TRUST_SERVER_CERTIFICATE", "yes")
    DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"
    DB_LOGGING_ENABLED = os.getenv("DB_LOGGING_ENABLED", "true").lower() == "true"
    LOGS_API_KEY = os.getenv("LOGS_API_KEY", "")
    KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "true").lower() == "true"
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
    KAFKA_TOPIC_PREDICTIONS = os.getenv("KAFKA_TOPIC_PREDICTIONS", "lung.predictions")
    KAFKA_CLIENT_ID = os.getenv("KAFKA_CLIENT_ID", "lung-api-producer")
    KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "")
    KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
    KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
    KAFKA_GROUP_DB = os.getenv("KAFKA_GROUP_DB", "lung-consumer-db")
    KAFKA_GROUP_MONITORING = os.getenv("KAFKA_GROUP_MONITORING", "lung-consumer-monitoring")
    KAFKA_GROUP_ANALYTICS = os.getenv("KAFKA_GROUP_ANALYTICS", "lung-consumer-analytics")
    KAFKA_GROUP_DOCTOR = os.getenv("KAFKA_GROUP_DOCTOR", "lung-consumer-doctor")
    KAFKA_GROUP_NOTIFICATIONS = os.getenv("KAFKA_GROUP_NOTIFICATIONS", "lung-consumer-notifications")

    MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"
    MLFLOW_STRICT = os.getenv("MLFLOW_STRICT", "false").lower() == "true"
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

    MLFLOW_MODEL_NAME_DENSENET_BINARY = os.getenv("MLFLOW_MODEL_NAME_DENSENET_BINARY", "lung-binary-densenet")
    MLFLOW_MODEL_NAME_EFFICIENTNET_BINARY = os.getenv("MLFLOW_MODEL_NAME_EFFICIENTNET_BINARY", "lung-binary-efficientnet")
    MLFLOW_MODEL_NAME_INCEPTION_BINARY = os.getenv("MLFLOW_MODEL_NAME_INCEPTION_BINARY", "lung-binary-inception")
    MLFLOW_MODEL_NAME_MOBILENET_BINARY = os.getenv("MLFLOW_MODEL_NAME_MOBILENET_BINARY", "lung-binary-mobilenet")
    MLFLOW_MODEL_NAME_DISEASES = os.getenv("MLFLOW_MODEL_NAME_DISEASES", "lung-diseases-densenet")
    MLFLOW_MODEL_NAME_SEGMENTATION = os.getenv("MLFLOW_MODEL_NAME_SEGMENTATION", "lung-segmentation-unet-xception")

    @staticmethod
    def get_metadata_path(model_dir: Path) -> Path:
        ''' Returns the path to the metadata.yaml inside a specific model folder '''
        return model_dir / "metadata.yaml"

    @staticmethod
    def get_database_url() -> str:
        user = quote_plus(AppConfig.DB_USER)
        password = quote_plus(AppConfig.DB_PASSWORD)
        driver = quote_plus(AppConfig.DB_DRIVER)
        host = AppConfig.DB_HOST
        port = AppConfig.DB_PORT.strip() if AppConfig.DB_PORT else ""

        server = host
        if port and "\\" not in host:
            server = f"{host}:{port}"

        return (
            f"mssql+pyodbc://{user}:{password}@{server}/{AppConfig.DB_NAME}"
            f"?driver={driver}"
            f"&Encrypt={AppConfig.DB_ENCRYPT}"
            f"&TrustServerCertificate={AppConfig.DB_TRUST_SERVER_CERTIFICATE}"
        )
