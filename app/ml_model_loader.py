import os
import joblib
import pandas as pd

from app.image_features import extract_image_features


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "artifacts", "image_optimizer_model.joblib")

_model_bundle = None


def load_recommendation_model():
    global _model_bundle

    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. Run 'python -m ml.train' first."
            )

        _model_bundle = joblib.load(MODEL_PATH)

    return _model_bundle


def predict_recommendation(image_bytes: bytes) -> dict:
    model_bundle = load_recommendation_model()

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    target_columns = model_bundle["target_columns"]

    features = extract_image_features(image_bytes)

    input_df = pd.DataFrame([[features[col] for col in feature_columns]], columns=feature_columns)

    prediction = model.predict(input_df)[0]

    result = dict(zip(target_columns, prediction))

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
        "model_path": MODEL_PATH,
    }