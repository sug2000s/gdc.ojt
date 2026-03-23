# ATI GDC - Assetization System Handover / Bàn giao hệ thống Assetization

베트남 GDC 인원 대상 Assetization 시스템 인수인계 자료
/ Handover materials for Vietnam GDC team
/ Tài liệu bàn giao cho đội GDC Việt Nam

## 일정 / Schedule / Lịch trình

- 기간 / Duration / Thời gian: 2주 / 2 weeks / 2 tuần (Week 1 ~ Week 2)
- 대상 / Target / Đối tượng: 베트남 GDC 팀 / Vietnam GDC Team / Đội GDC Việt Nam

## 시스템 구성 / System Overview / Tổng quan hệ thống (8개 레포 / 8 repos / 8 repo, ~44,000줄 / lines / dòng)

| # | 서비스 / Service / Dịch vụ | 스택 / Stack / Công nghệ | 역할 / Role / Vai trò |
|---|--------|------|------|
| 1 | assetization_mobile | React 19 + TS + Vite | 프론트엔드 / Frontend / Giao diện |
| 2 | assetization_auth | Python + FastAPI | 인증 게이트웨이 / Auth Gateway / Cổng xác thực |
| 3 | assetization_orchestrator | Python + FastAPI + LangGraph | AI 채팅/검색 엔진 / AI Chat/Search / Trò chuyện/Tìm kiếm AI |
| 4 | assetization_mcp | Python + FastAPI + MCP | 리스크 분석 / Risk Analysis / Phân tích rủi ro |
| 5 | assetization_datacenter | Python + FastAPI | 히스토리/태그/컬렉션 / History/Tag/Collection / Lịch sử/Tag/Bộ sưu tập |
| 6 | assetization_api | Python + FastAPI + LangChain | 레거시 API / Legacy API / API cũ |
| 7 | keylook-officeplus | Python + FastAPI + FlagEmbedding | 하이브리드 검색 엔진 / Hybrid Search Engine / Công cụ tìm kiếm lai |
| 8 | keylook_script | Shell | Docker 래퍼 / Docker Wrapper / Trình bọc Docker |

## 폴더 구조 / Folder Structure / Cấu trúc thư mục

```
ati-gdc/
├── docs/
│   ├── system-overview/
│   │   ├── 00-system-map.md           # 아키텍처 / Architecture / Kiến trúc
│   │   └── 01-service-catalog.md      # 소스 경로 / Source Paths / Đường dẫn mã nguồn
│   ├── architecture/
│   │   ├── 01-api-endpoints.md        # API 맵 / API Map / Bản đồ API (80+)
│   │   ├── 02-database-schema.md      # DB 스키마 / DB Schema / Lược đồ DB
│   │   └── 03-infra-and-config.md     # 인프라 / Infra & Deploy / Hạ tầng & Triển khai
│   └── api-reference/                 # (추가 예정 / TBD / Sẽ bổ sung)
├── labs/
│   ├── week1/                         # 1주차 / Week 1 / Tuần 1
│   └── week2/                         # 2주차 / Week 2 / Tuần 2
└── resources/                         # 참고 자료 / References / Tài liệu tham khảo
```

## 문서 가이드 / Document Guide / Hướng dẫn tài liệu

### 시작하기 / Getting Started / Bắt đầu
1. [00-system-map.md](docs/system-overview/00-system-map.md) - 전체 그림 / Big Picture / Tổng quan
2. [01-service-catalog.md](docs/system-overview/01-service-catalog.md) - 소스 경로 / Source Paths / Đường dẫn mã nguồn

### 깊이 있는 이해 / Deep Dive / Tìm hiểu sâu
3. [01-api-endpoints.md](docs/architecture/01-api-endpoints.md) - API 목록 / API List / Danh sách API
4. [02-database-schema.md](docs/architecture/02-database-schema.md) - 데이터 모델 / Data Model / Mô hình dữ liệu
5. [03-infra-and-config.md](docs/architecture/03-infra-and-config.md) - 인프라 & 배포 / Infra & Deploy / Hạ tầng & Triển khai

## Week 1 - 시스템 이해 / Understanding / Hiểu hệ thống

- 시스템 개요 / System Overview / Tổng quan hệ thống
- 아키텍처 및 데이터 흐름 / Architecture & Data Flow / Kiến trúc & Luồng dữ liệu
- 개발 환경 셋업 / Dev Environment Setup / Cài đặt môi trường
- 핵심 모듈 코드 리딩 / Core Module Code Reading / Đọc code module chính

## Week 2 - 실전 및 운영 / Practice & Ops / Thực hành & Vận hành

- 기능 개발 실습 / Feature Development / Phát triển tính năng
- 디버깅 & 트러블슈팅 / Debugging & Troubleshooting / Gỡ lỗi & Xử lý sự cố
- 배포 프로세스 / Deployment Process / Quy trình triển khai
- 운영 모니터링 & 대응 / Monitoring & Response / Giám sát & Ứng phó
