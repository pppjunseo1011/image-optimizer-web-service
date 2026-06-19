import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from app.config import MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_URI, REGISTERED_MODEL_NAME
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
from ml.model_promoter import promote_if_better

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "image_optimize_train.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "image_optimizer_model.joblib")

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI") or MLFLOW_TRACKING_URI
EXPERIMENT_NAME = "image-optimization-recommendation"
# REGISTERED_MODEL_NAME = "image-optimizer-model"

FEATURE_COLUMNS = [
    "width",
    "height",
    "file_size_kb",
    "pixel_count",
    "aspect_ratio",
    "format_code",
    "has_alpha",
]

TARGET_COLUMNS = [
    "recommended_action",
    "recommended_format",
    "recommended_quality",
]


def train_model():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    with mlflow.start_run(run_name="RandomForestClassifier"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("row_count", len(df))
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("target_count", len(TARGET_COLUMNS))

        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_accuracy = (y_train.to_numpy() == train_preds).all(axis=1).mean()
        test_accuracy = (y_test.to_numpy() == test_preds).all(axis=1).mean()

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        joblib.dump(
            {
                "model": model,
                "feature_columns": FEATURE_COLUMNS,
                "target_columns": TARGET_COLUMNS,
            },
            MODEL_PATH,
        )

        mlflow.log_artifact(DATA_PATH)
        mlflow.log_artifact(MODEL_PATH)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        
        client = MlflowClient()

        latest_versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        latest_version = max(
            latest_versions,
            key=lambda version: int(version.version)
        ).version

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="challenger",
            version=str(latest_version)
        )

        print(f"challenger alias set to version: {latest_version}")

        promote_if_better(
            new_version=latest_version,
            new_test_accuracy=test_accuracy,
        )

        print(f"Model saved to: {MODEL_PATH}")
        print(f"train_accuracy: {train_accuracy:.4f}")
        print(f"test_accuracy: {test_accuracy:.4f}")
        print("MLflow run logged successfully")


if __name__ == "__main__":
    train_model()