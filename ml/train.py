import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "image_optimize_train.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "image_optimizer_model.joblib")

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

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_accuracy = (y_train.to_numpy() == train_preds).all(axis=1).mean()
    test_accuracy = (y_test.to_numpy() == test_preds).all(axis=1).mean()

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
        },
        MODEL_PATH,
    )

    print(f"Model saved to: {MODEL_PATH}")
    print(f"train_accuracy: {train_accuracy:.4f}")
    print(f"test_accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train_model()