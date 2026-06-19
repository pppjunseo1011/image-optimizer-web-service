import mlflow
from mlflow.tracking import MlflowClient

from app.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTRY_URI,
    REGISTERED_MODEL_NAME,
)


def get_champion_test_accuracy(client: MlflowClient) -> float:
    """
    현재 champion 모델의 test_accuracy를 가져온다.
    champion이 없으면 -1.0을 반환한다.
    """
    try:
        champion = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            "champion",
        )
        champion_run = client.get_run(champion.run_id)
        return float(champion_run.data.metrics.get("test_accuracy", -1.0))

    except Exception:
        return -1.0


def promote_if_better(new_version: str, new_test_accuracy: float):
    """
    신규 모델의 성능이 현재 champion보다 좋으면 champion으로 승격한다.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)

    client = MlflowClient()

    current_champion_acc = get_champion_test_accuracy(client)

    print(f"[PROMOTION] current champion test_accuracy = {current_champion_acc}")
    print(f"[PROMOTION] new candidate test_accuracy = {new_test_accuracy}")

    if new_test_accuracy > current_champion_acc:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=str(new_version),
        )
        print(f"[PROMOTION] version {new_version} promoted to champion")
    else:
        print("[PROMOTION] champion unchanged")