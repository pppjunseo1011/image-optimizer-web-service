import logging

from pydantic import BaseModel

from app.config import LOW_CONFIDENCE_THRESHOLD
from app.prediction_logger import save_prediction_log
from app.feedback import save_feedback
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from app.ml_model_loader import predict_recommendation

from app.image_utils import (
    resize_image,
    compress_image,
    convert_image,
    EXT_MAP,
    MEDIA_TYPE_MAP,
)

app = FastAPI(
    title="Image Optimizer Web Service",
    description="이미지 업로드, 리사이즈, 압축, 포맷 변환을 지원하는 FastAPI 기반 웹 서비스",
    version="1.0.0",
)

logger = logging.getLogger(__name__)

class FeedbackRequest(BaseModel):
    recommended_action: str
    user_selected_action: str
    recommended_format: str
    user_selected_format: str
    score: float | None = None
    comment: str = ""

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8" />
        <title>Image Optimizer Web Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                background: #f7f9fc;
                color: #222;
            }
            h1 {
                color: #1f4f9f;
            }
            .card {
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            input, select, button {
                margin: 6px 0;
                padding: 8px;
            }
            button {
                background: #1f4f9f;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
            }
            button:hover {
                background: #163b78;
            }
        </style>
    </head>
    <body>
        <h1>Image Optimizer Web Service</h1>
        <p>이미지 업로드, 리사이즈, 압축, 포맷 변환을 수행하는 웹 서비스입니다.</p>

        <div class="card">
            <h2>1. 이미지 리사이즈</h2>
            <form action="/resize" method="post" enctype="multipart/form-data" target="_blank">
                <input type="file" name="file" accept="image/*" required><br>
                Width: <input type="number" name="width" value="512" required><br>
                Height: <input type="number" name="height" value="512" required><br>
                Format:
                <select name="output_format">
                    <option value="JPEG">JPG</option>
                    <option value="PNG">PNG</option>
                    <option value="WEBP">WEBP</option>
                </select><br>
                <button type="submit">Resize</button>
            </form>
        </div>

        <div class="card">
            <h2>2. 이미지 압축</h2>
            <form action="/compress" method="post" enctype="multipart/form-data" target="_blank">
                <input type="file" name="file" accept="image/*" required><br>
                Quality: <input type="number" name="quality" value="70" min="1" max="100" required><br>
                Format:
                <select name="output_format">
                    <option value="JPEG">JPG</option>
                    <option value="PNG">PNG</option>
                    <option value="WEBP">WEBP</option>
                </select><br>
                <button type="submit">Compress</button>
            </form>
        </div>

        <div class="card">
            <h2>3. 이미지 포맷 변환</h2>
            <form action="/convert" method="post" enctype="multipart/form-data" target="_blank">
                <input type="file" name="file" accept="image/*" required><br>
                Convert to:
                <select name="output_format">
                    <option value="JPEG">JPG</option>
                    <option value="PNG">PNG</option>
                    <option value="WEBP">WEBP</option>
                </select><br>
                <button type="submit">Convert</button>
            </form>
        </div>

        <div class="card">
            <h2>서비스 상태 확인</h2>
            <p><a href="/health" target="_blank">/health</a></p>
            <p><a href="/docs" target="_blank">API 문서 보기</a></p>
        </div>
    </body>
    </html>
    """


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Image Optimizer Web Service",
    }


@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...),
    output_format: str = Form("JPEG"),
):
    file_bytes = await file.read()
    output, fmt = resize_image(file_bytes, width, height, output_format)

    extension = EXT_MAP[fmt]
    media_type = MEDIA_TYPE_MAP[fmt]

    return StreamingResponse(
        output,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="resized_image.{extension}"'
        },
    )


@app.post("/compress")
async def compress_endpoint(
    file: UploadFile = File(...),
    quality: int = Form(70),
    output_format: str = Form("JPEG"),
):
    file_bytes = await file.read()
    output, fmt = compress_image(file_bytes, quality, output_format)

    extension = EXT_MAP[fmt]
    media_type = MEDIA_TYPE_MAP[fmt]

    return StreamingResponse(
        output,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="compressed_image.{extension}"'
        },
    )


@app.post("/convert")
async def convert_endpoint(
    file: UploadFile = File(...),
    output_format: str = Form("WEBP"),
):
    file_bytes = await file.read()
    output, fmt = convert_image(file_bytes, output_format)

    extension = EXT_MAP[fmt]
    media_type = MEDIA_TYPE_MAP[fmt]

    return StreamingResponse(
        output,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="converted_image.{extension}"'
        },
    )
    
@app.post("/recommend")
async def recommend_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_recommendation(image_bytes)

        features = result["features"]
        model_info = result.get("model_info", {})

        save_prediction_log(
            width=features["width"],
            height=features["height"],
            file_size_kb=features["file_size_kb"],
            recommended_action=result["recommended_action"],
            recommended_format=result["recommended_format"],
            recommended_quality=result["recommended_quality"],
            score=result["score"],
            model_uri=model_info.get("model_uri", "unknown"),
            run_id=model_info.get("run_id", "unknown"),
        )

        if result["score"] is not None and result["score"] < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                f"LOW CONFIDENCE /recommend | score={result['score']} "
                f"action={result['recommended_action']} "
                f"format={result['recommended_format']}"
            )

        logger.info(
            f"OK /recommend | action={result['recommended_action']} "
            f"format={result['recommended_format']} score={result['score']}"
        )

        return result

    except Exception as e:
        logger.exception(f"FAIL /recommend | error={type(e).__name__}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/feedback")
async def feedback(payload: FeedbackRequest):
    try:
        save_feedback(
            recommended_action=payload.recommended_action,
            user_selected_action=payload.user_selected_action,
            recommended_format=payload.recommended_format,
            user_selected_format=payload.user_selected_format,
            score=payload.score,
            comment=payload.comment,
        )

        logger.info(
            f"OK /feedback | recommended_action={payload.recommended_action} "
            f"user_selected_action={payload.user_selected_action}"
        )

        return {"status": "feedback saved"}

    except Exception as e:
        logger.exception(f"FAIL /feedback | error={type(e).__name__}: {e}")
        raise HTTPException(status_code=400, detail=str(e))