# ML 기반 이미지 최적화 추천 서비스

이미지를 업로드하면 이미지의 크기, 파일 용량, 포맷 등의 특징을 분석하고, 머신러닝 모델을 이용해 적절한 이미지 최적화 방식을 추천하는 FastAPI 기반 웹 서비스입니다.

본 프로젝트는 단순 이미지 처리 기능뿐만 아니라, MLflow를 활용한 모델 관리, GitHub Actions 기반 자동화, Docker 기반 실행 환경, Render 배포, 예측 로그 및 모니터링 기능까지 포함한 MLOps 실습 프로젝트입니다.

---

## 1. 프로젝트 주요 기능

* 이미지 업로드
* 이미지 리사이즈
* 이미지 압축
* 이미지 포맷 변환
* ML 기반 이미지 최적화 추천
* MLflow 기반 모델 학습 기록 및 버전 관리
* `champion` alias 기반 운영 모델 로딩
* 예측 결과 로그 저장
* 사용자 피드백 저장
* Streamlit 기반 모니터링 대시보드
* Docker 기반 컨테이너 실행
* GitHub Actions 기반 테스트 및 모델 학습 자동화

---

## 2. 사용 기술

| 구분      | 사용 기술                                  |
| ------- | -------------------------------------- |
| 웹 프레임워크 | FastAPI, Uvicorn                       |
| 이미지 처리  | Pillow                                 |
| 머신러닝    | scikit-learn, pandas, joblib           |
| 모델 관리   | MLflow Tracking, MLflow Model Registry |
| 테스트     | pytest, httpx                          |
| 자동화     | GitHub Actions                         |
| 컨테이너    | Docker                                 |
| 배포      | Render                                 |
| 모니터링    | Streamlit, CSV 로그                      |

---

## 3. 프로젝트 구조

```text
image-optimizer-web-service/
├── app/
│   ├── main.py
│   ├── image_features.py
│   ├── ml_model_loader.py
│   ├── prediction_logger.py
│   └── feedback.py
├── ml/
│   ├── train.py
│   ├── model_promoter.py
│   ├── data/
│   │   └── image_optimize_train.csv
│   └── artifacts/
├── tests/
├── dashboard.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .github/
    └── workflows/
```

---

## 4. 실행 방법

### 4.1 저장소 복제

```bash
git clone https://github.com/pppjunseo1011/image-optimizer-web-service.git
cd image-optimizer-web-service
```

### 4.2 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 4.3 FastAPI 서버 실행

```bash
uvicorn app.main:app --reload
```

실행 후 아래 주소로 접속합니다.

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

`/docs`에서는 API를 직접 테스트할 수 있습니다.

---

## 5. 주요 API 사용 방법

### 5.1 서비스 상태 확인

```http
GET /health
```

서비스가 정상적으로 실행 중인지 확인합니다.

---

### 5.2 이미지 최적화 추천

```http
POST /recommend
```

이미지 파일을 업로드하면 ML 모델이 이미지 특징을 분석하여 최적화 방식을 추천합니다.

응답 예시는 다음과 같습니다.

```json
{
  "features": {
    "width": 800,
    "height": 600,
    "file_size_kb": 120.5,
    "pixel_count": 480000,
    "aspect_ratio": 1.33,
    "format_code": 0,
    "has_alpha": 0
  },
  "recommendation": {
    "recommended_action": "compress",
    "recommended_format": "WEBP",
    "recommended_quality": 80
  },
  "score": 0.85,
  "model_info": {
    "model_uri": "models:/image-optimizer-model@champion",
    "run_id": "...",
    "model_type": "RandomForestClassifier",
    "test_accuracy": 0.625
  }
}
```

---

### 5.3 사용자 피드백 저장

```http
POST /feedback
```

추천 결과에 대한 사용자 피드백을 저장합니다.
저장된 피드백은 이후 모델 개선이나 재학습 데이터로 활용할 수 있습니다.

---

## 6. 모델 학습 방법

ML 모델을 다시 학습하려면 아래 명령을 실행합니다.

```bash
python -m ml.train
```

학습 과정에서는 다음 작업이 수행됩니다.

```text
학습 데이터 불러오기
→ 이미지 특징 기반 모델 학습
→ train/test accuracy 계산
→ 모델 artifact 저장
→ MLflow에 실험 결과 기록
→ MLflow Model Registry에 모델 등록
```

---

## 7. MLflow 사용 방법

MLflow 서버를 실행합니다.

```bash
mlflow server --host 0.0.0.0 --port 9999 --backend-store-uri sqlite:///mlflow.db
```

MLflow UI는 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:9999
```

MLflow에는 다음 정보가 기록됩니다.

* 모델 종류
* 학습 파라미터
* train accuracy
* test accuracy
* 학습 데이터 artifact
* 모델 artifact
* 등록 모델 버전

서비스는 다음 모델 URI를 통해 운영 모델을 불러옵니다.

```text
models:/image-optimizer-model@champion
```

따라서 새로운 모델이 등록되더라도 서비스 코드를 직접 수정하지 않고, `champion` alias만 변경하여 운영 모델을 교체할 수 있습니다.

---

## 8. 모델 교체 방식

새로운 모델이 학습되면 MLflow Model Registry에 새 버전으로 등록됩니다.

모델 교체 흐름은 다음과 같습니다.

```text
신규 모델 학습
→ MLflow Model Registry에 새 버전 등록
→ 기존 champion 모델과 성능 비교
→ 신규 모델 성능이 더 좋으면 champion으로 승격
→ 서비스는 새로운 champion 모델 사용
```

모델 승격 로직은 아래 파일에 구현되어 있습니다.

```text
ml/model_promoter.py
```

---

## 9. Docker 실행 방법

Docker 이미지를 빌드합니다.

```bash
docker build -t image-optimizer-mlops:local .
```

컨테이너를 실행합니다.

```bash
docker run --rm -p 10000:10000 image-optimizer-mlops:local
```

실행 후 아래 주소로 접속합니다.

```text
http://127.0.0.1:10000/docs
```

---

## 10. 모니터링 대시보드 실행

서비스의 예측 로그와 사용자 피드백 로그는 아래 파일에 저장됩니다.

```text
logs/predictions.csv
logs/feedback.csv
```

Streamlit 대시보드를 실행합니다.

```bash
streamlit run dashboard.py
```

실행 후 아래 주소로 접속합니다.

```text
http://localhost:8501
```

대시보드에서는 다음 정보를 확인할 수 있습니다.

* 전체 예측 요청 수
* 평균 예측 신뢰도
* 낮은 confidence 요청 수
* 최근 예측 결과
* 사용자 피드백 수
* 추천 결과와 사용자 선택이 다른 사례

---

## 11. 테스트 실행

전체 테스트를 실행합니다.

```bash
pytest
```

자세한 테스트 결과를 확인하려면 아래 명령을 사용합니다.

```bash
pytest -v
```

---

## 12. 배포 주소

Render 배포 주소:

```text
https://image-optimizer-web-service.onrender.com
```

API 문서:

```text
https://image-optimizer-web-service.onrender.com/docs
```

GitHub 저장소:

```text
https://github.com/pppjunseo1011/image-optimizer-web-service
```

---

## 13. 커밋하지 않는 파일

아래 파일과 폴더는 실행 중 자동으로 생성되는 결과물이므로 Git에 커밋하지 않습니다.

```text
mlflow.db
mlruns/
mlartifacts/
logs/
ml/artifacts/
__pycache__/
.pytest_cache/
```

이 항목들은 `.gitignore`에 포함하는 것을 권장합니다.
