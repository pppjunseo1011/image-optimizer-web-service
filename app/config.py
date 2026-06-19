import os


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI") or MLFLOW_TRACKING_URI

REGISTERED_MODEL_NAME = "image-optimizer-model"
MODEL_URI = os.getenv("MODEL_URI") or "models:/image-optimizer-model@champion"

LOW_CONFIDENCE_THRESHOLD = 0.65