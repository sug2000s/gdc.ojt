# 인프라 및 환경 설정 / Infrastructure & Config / Hạ tầng & Cấu hình

## 환경별 구성 / Environment Config / Cấu hình theo môi trường

| 항목 / Item / Hạng mục | Dev | QA | Prod |
|------|-----|-----|------|
| SSO | qsso.lgcns.com | qsso.lgcns.com | sso.lgcns.com |
| EMS | test-moffice.lgcns.com | test-moffice.lgcns.com | moffice.lgcns.com |
| MSSQL CommonDB | localhost:1433 | LCNQAWSDB01.corp.lgcns.com | PMI01.corp.lgcns.com |
| KeyLook | qaxgpu.lgcns.com | qaxgpu.lgcns.com | axgpu.lgcns.com |
| Redis | standalone (localhost) | Cluster | Cluster |
| Kafka | localhost:9092 | AWS MSK | AWS MSK |

## 서비스별 포트 / Ports by Service / Cổng theo dịch vụ

| 서비스 / Service / Dịch vụ | Dev | QA | Prod |
|--------|-----|-----|------|
| assetization_auth | 9090 | 9091 | 8080 |
| assetization_orchestrator | 8000 | 80 | 80 |
| assetization_mcp | 8080 | 9090 | 9090 |
| assetization_datacenter | 80 | 80 | 80 |
| assetization_api | 8000 | 8000 | 8000 |
| keylook-officeplus | 9090 | 8001-8006 | 8001-8006 |

## Docker 구성 / Docker Config / Cấu hình Docker

### assetization_auth
- **Base**: `sug2000s/python:3.13-slim`
- **Compose**: LocalStack S3 + Nginx reverse proxy
- **Workers**: 4 (prod)

### assetization_orchestrator
- **Base**: ECR `python:3.13-slim`
- **Gunicorn**: 4 workers, 300s timeout, port 80
- **Volumes**: logs, upload files / 로그, 업로드 파일 / log, file tải lên

### assetization_mcp
- **Base**: `python:3.13-slim`
- **Compose**: Zookeeper + Kafka + MCP RAG App
- **Nginx**: reverse proxy

### assetization_datacenter
- **Base**: `python:3.9.11`
- **특이사항 / Note / Ghi chú**: MeCab C++ 빌드 포함 / Includes MeCab C++ build / Bao gồm build MeCab C++
- **Gunicorn**: 4 workers, port 80

### assetization_api
- **Base**: ECR `python:3.13-slim`
- **특이사항 / Note / Ghi chú**: ODBC Driver 18 포함 / Includes ODBC Driver 18 / Bao gồm ODBC Driver 18
- **Compose**: API + Echo + Nginx

### keylook-officeplus
- **Base**: `sug2000s/keylook-officeplus:latest`
- **Workers**: 6개 Uvicorn / 6 Uvicorn instances / 6 tiến trình Uvicorn (ports 8001-8006)
- **Healthcheck**: curl /health

## 공통 환경변수 / Common Env Vars / Biến môi trường chung

### 모든 Python 백엔드 / All Python Backends / Tất cả backend Python
```
APP_ENV=dev|qa|prod       # 환경 선택 / Env selection / Chọn môi trường
APP_WORKER=4              # Gunicorn 워커 수 / Worker count / Số worker
LOG_LEVEL=DEBUG|INFO      # 로그 레벨 / Log level / Mức log
TZ=Asia/Seoul             # 타임존 / Timezone / Múi giờ
```

### Redis 관련 / Redis Related / Liên quan Redis
```
REDIS_HOST, REDIS_PORT, REDIS_DB
REDIS_CLUSTER_MODE=true|false
REDIS_PASSWORD
```

### MSSQL 관련 / MSSQL Related / Liên quan MSSQL
```
DB_COMMON_HOST, DB_COMMON_PORT
DB_COMMON_NAME, DB_COMMON_USER, DB_COMMON_PASSWORD
```

### Azure OpenAI 관련 / Azure OpenAI Related / Liên quan Azure OpenAI
```
AZURE_ENDPOINT            # API 엔드포인트 / API endpoint / Điểm cuối API
AZURE_API_KEY             # API 키 / API key / Khóa API
AZURE_API_VERSION         # 2024-12-01-preview
AZURE_DEPLOYMENT          # gpt-4.1 / gpt-4o
```

### Kafka 관련 / Kafka Related / Liên quan Kafka
```
KAFKA_BOOTSTRAP_SERVERS
KAFKA_TOPIC_ACCESS_LOG=access_logging
```

## 배포 스크립트 / Deploy Scripts / Script triển khai

| 서비스 / Service / Dịch vụ | 스크립트 / Script / Script | 설명 / Description / Mô tả |
|--------|---------|------|
| assetization_auth | `run.sh` | dev/qa/prod 모드 실행 / Mode execution / Chạy theo chế độ |
| assetization_auth | `scripts/copy-front.sh` | 프론트엔드 빌드 복사 / Frontend build copy / Sao chép build giao diện |
| assetization_auth | `scripts/deploy-frontend.sh` | S3 배포 / S3 deploy / Triển khai S3 |
| assetization_mobile | `deploy.sh` | QA 빌드 + git 배포 / QA build + git deploy / Build QA + triển khai git |
| assetization_mcp | `scripts/quick-deploy.sh` | 빠른 프로덕션 배포 / Quick prod deploy / Triển khai prod nhanh |
| assetization_mcp | `scripts/build-and-push.sh` | ECR 빌드/푸시 / ECR build/push / Build/push ECR |
| keylook_script | `run.sh` | Docker 실행 래퍼 / Docker run wrapper / Trình bọc chạy Docker |
