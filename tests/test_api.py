from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def create_test_image(
    image_format: str = "PNG",
    size: tuple[int, int] = (100, 100),
) -> BytesIO:
    image = Image.new("RGB", size, color="blue")
    image_bytes = BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)
    return image_bytes


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Image Optimizer Web Service"


def test_resize_image():
    image_file = create_test_image()

    response = client.post(
        "/resize",
        files={"file": ("test.png", image_file, "image/png")},
        data={
            "width": "50",
            "height": "40",
            "output_format": "PNG",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    result_image = Image.open(BytesIO(response.content))
    assert result_image.size == (50, 40)


def test_compress_image():
    image_file = create_test_image("JPEG")

    response = client.post(
        "/compress",
        files={"file": ("test.jpg", image_file, "image/jpeg")},
        data={
            "quality": "60",
            "output_format": "JPEG",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_convert_image_to_webp():
    image_file = create_test_image("PNG")

    response = client.post(
        "/convert",
        files={"file": ("test.png", image_file, "image/png")},
        data={
            "output_format": "WEBP",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"

    result_image = Image.open(BytesIO(response.content))
    assert result_image.format == "WEBP"


def test_invalid_format():
    image_file = create_test_image()

    response = client.post(
        "/convert",
        files={"file": ("test.png", image_file, "image/png")},
        data={
            "output_format": "GIF",
        },
    )

    assert response.status_code == 400