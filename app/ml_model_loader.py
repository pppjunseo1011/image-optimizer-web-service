import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from app.config import MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_URI, MODEL_URI
from app.image_features import extract_image_features


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

_model = None


def load_recommendation_model():
    global _model

    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)

    return _model


def get_model_info():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)

    try:
        info = mlflow.models.get_model_info(MODEL_URI)
        run = MlflowClient().get_run(info.run_id)

        return {
            "model_uri": MODEL_URI,
            "run_id": info.run_id,
            "model_type": run.data.params.get("model_type"),
            "test_accuracy": run.data.metrics.get("test_accuracy"),
        }

    except Exception:
        return {
            "model_uri": MODEL_URI,
            "run_id": "unknown",
            "model_type": None,
            "test_accuracy": None,
        }


def predict_recommendation(image_bytes: bytes) -> dict:
    model = load_recommendation_model()

    features = extract_image_features(image_bytes)

    input_df = pd.DataFrame(
        [[features[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(input_df)[0]
    result = dict(zip(TARGET_COLUMNS, prediction))

    score = None

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_df)

        if isinstance(probas, list):
            max_probs = [float(p.max()) for p in probas]
            score = sum(max_probs) / len(max_probs)
        else:
            score = float(probas.max())

    return {
        "features": features,
        "recommended_action": result["recommended_action"],
        "recommended_format": result["recommended_format"],
        "recommended_quality": int(result["recommended_quality"]),
        "score": round(score, 4) if score is not None else None,
        "model_info": get_model_info(),
    }