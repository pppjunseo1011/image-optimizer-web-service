import os
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from ml.train import MODEL_PATH, train_model


def make_test_image_bytes():
    image = Image.new("RGB", (800, 600), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_recommend_api_returns_prediction():
    # 테스트 환경에 모델 파일이 없으면 먼저 학습을 수행한다.
    if not os.path.exists(MODEL_PATH):
        train_model()

    client = TestClient(app)

    image_bytes = make_test_image_bytes()
    files = {
        "file": ("test.jpg", image_bytes, "image/jpeg")
    }

    response = client.post("/recommend", files=files)

    assert response.status_code == 200

    data = response.json()

    assert "features" in data
    assert data["features"]["width"] == 800
    assert data["features"]["height"] == 600

    assert data["recommended_action"] in ["compress", "convert", "resize"]
    assert data["recommended_format"] in ["WEBP", "PNG"]
    assert isinstance(data["recommended_quality"], int)

    if data["score"] is not None:
        assert 0 <= data["score"] <= 1