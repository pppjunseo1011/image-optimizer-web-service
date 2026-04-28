from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Image Optimizer Web Service",
    description="이미지 업로드, 리사이즈, 압축, 포맷 변환을 지원하는 웹 서비스",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Image Optimizer Web Service</h1>
    <p>이미지 업로드, 리사이즈, 압축, 포맷 변환을 지원하는 웹 서비스입니다.</p>
    <p><a href="/health">Health Check</a></p>
    <p><a href="/docs">API Docs</a></p>
    """


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Image Optimizer Web Service",
    }