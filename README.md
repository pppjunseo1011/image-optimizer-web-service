# Image Optimizer Web Service

FastAPI 기반 이미지 변환/압축 웹 서비스입니다.  
사용자는 이미지를 업로드하여 리사이즈, 압축, 포맷 변환 기능을 사용할 수 있습니다.

## 주요 기능

- 이미지 업로드
- 이미지 리사이즈
- 이미지 압축
- 이미지 포맷 변환
- `/health` 상태 확인
- FastAPI 자동 API 문서 제공

## 기술 스택

- Python 3.11
- FastAPI
- Pillow
- pytest
- Docker
- GitHub Actions
- Render

## 로컬 실행 방법

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload