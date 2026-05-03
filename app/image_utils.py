from io import BytesIO
from typing import Tuple

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError


FORMAT_MAP = {
    "JPG": "JPEG",
    "JPEG": "JPEG",
    "PNG": "PNG",
    "WEBP": "WEBP",
}

EXT_MAP = {
    "JPEG": "jpg",
    "PNG": "png",
    "WEBP": "webp",
}

MEDIA_TYPE_MAP = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def normalize_format(output_format: str) -> str:
    fmt = output_format.upper().strip()

    if fmt not in FORMAT_MAP:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 이미지 포맷입니다. JPG, PNG, WEBP 중 하나를 선택하세요.",
        )

    return FORMAT_MAP[fmt]


def load_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()
        return image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="올바른 이미지 파일이 아닙니다.")


def prepare_image_for_format(image: Image.Image, output_format: str) -> Image.Image:
    if output_format == "JPEG" and image.mode in ("RGBA", "P"):
        return image.convert("RGB")
    return image


def save_image_to_bytes(
    image: Image.Image,
    output_format: str,
    quality: int = 80,
) -> BytesIO:
    output = BytesIO()
    image = prepare_image_for_format(image, output_format)

    save_options = {}

    if output_format in ("JPEG", "WEBP"):
        save_options["quality"] = quality
        save_options["optimize"] = True

    if output_format == "PNG":
        save_options["optimize"] = True

    image.save(output, format=output_format, **save_options)
    output.seek(0)
    return output


def resize_image(
    file_bytes: bytes,
    width: int,
    height: int,
    output_format: str,
) -> Tuple[BytesIO, str]:
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="가로와 세로 크기는 1 이상이어야 합니다.")

    fmt = normalize_format(output_format)
    image = load_image(file_bytes)
    
    # 의도적 버그: width와 height 순서를 반대로 입력
    resized = image.resize((height, width))

    output = save_image_to_bytes(resized, fmt)
    return output, fmt


def compress_image(
    file_bytes: bytes,
    quality: int,
    output_format: str,
) -> Tuple[BytesIO, str]:
    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="quality는 1부터 100 사이여야 합니다.")

    fmt = normalize_format(output_format)
    image = load_image(file_bytes)

    output = save_image_to_bytes(image, fmt, quality=quality)
    return output, fmt


def convert_image(
    file_bytes: bytes,
    output_format: str,
) -> Tuple[BytesIO, str]:
    fmt = normalize_format(output_format)
    image = load_image(file_bytes)

    output = save_image_to_bytes(image, fmt)
    return output, fmt