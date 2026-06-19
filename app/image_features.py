from io import BytesIO
from PIL import Image


FORMAT_CODE = {
    "JPEG": 0,
    "PNG": 1,
    "WEBP": 2,
    "BMP": 3,
    "GIF": 4,
    "UNKNOWN": -1,
}


def extract_image_features(image_bytes: bytes) -> dict:
    """
    업로드된 이미지에서 ML 모델 입력에 사용할 특징을 추출한다.
    """
    image_file = BytesIO(image_bytes)

    with Image.open(image_file) as image:
        width, height = image.size
        image_format = image.format or "UNKNOWN"
        mode = image.mode

    file_size_kb = len(image_bytes) / 1024
    pixel_count = width * height
    aspect_ratio = width / height if height != 0 else 0

    return {
        "width": width,
        "height": height,
        "file_size_kb": round(file_size_kb, 2),
        "pixel_count": pixel_count,
        "aspect_ratio": round(aspect_ratio, 3),
        "format": image_format,
        "format_code": FORMAT_CODE.get(image_format, -1),
        "mode": mode,
        "has_alpha": 1 if mode in ["RGBA", "LA"] else 0,
    }