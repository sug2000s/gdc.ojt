# ATI GDC - Assetization System Handover

베트남 GDC 인원 대상 Assetization 시스템 인수인계 자료

## 일정

- 기간: 2주 (Week 1 ~ Week 2)
- 대상: 베트남 GDC 팀

## 시스템 구성 (9개 레포지토리, ~56,000줄)

| # | 서비스 | 스택 | 역할 |
|---|--------|------|------|
| 1 | assetization_mobile | React 19 + TS + Vite | 모바일 프론트엔드 |
| 2 | assetization_authsso | Java + Spring Boot + JSP | PC 웹 프론트엔드 |
| 3 | assetization_auth | Python + FastAPI | 인증 게이트웨이 |
| 4 | assetization_orchestrator | Python + FastAPI + LangGraph | AI 채팅/검색 엔진 |
| 5 | assetization_mcp | Python + FastAPI + MCP | 리스크 분석 |
| 6 | assetization_datacenter | Python + FastAPI | 히스토리/태그/컬렉션 |
| 7 | assetization_api | Python + FastAPI + LangChain | 레거시 API |
| 8 | keylook-officeplus | Python + FastAPI + FlagEmbedding | 하이브리드 검색 엔진 |
| 9 | keylook_script | Shell | Docker 래퍼 |

## 폴더 구조

```
ati-gdc/
├── docs/
│   ├── system-overview/
│   │   ├── 00-system-map.md           # 전체 아키텍처 다이어그램 & 데이터 흐름
│   │   └── 01-service-catalog.md      # 서비스별 핵심 파일 & 소스 경로 총정리
│   ├── architecture/
│   │   ├── 01-api-endpoints.md        # 전체 API 엔드포인트 맵 (80+ endpoints)
│   │   ├── 02-database-schema.md      # DB 스키마 (PostgreSQL/MSSQL/Redis/ES)
│   │   └── 03-infra-and-config.md     # 인프라, 환경변수, Docker, 배포
│   └── api-reference/                 # (추가 예정)
├── labs/
│   ├── week1/                         # 1주차 실습 (추가 예정)
│   └── week2/                         # 2주차 실습 (추가 예정)
└── resources/                         # 참고 자료 (추가 예정)
```

## 문서 가이드

### 시작하기
1. [00-system-map.md](docs/system-overview/00-system-map.md) - 전체 그림 파악
2. [01-service-catalog.md](docs/system-overview/01-service-catalog.md) - 서비스별 소스 경로

### 깊이 있는 이해
3. [01-api-endpoints.md](docs/architecture/01-api-endpoints.md) - 전체 API 목록
4. [02-database-schema.md](docs/architecture/02-database-schema.md) - 데이터 모델
5. [03-infra-and-config.md](docs/architecture/03-infra-and-config.md) - 인프라 & 배포

## Week 1 - 시스템 이해

- 시스템 개요 및 비즈니스 컨텍스트
- 아키텍처 및 데이터 흐름
- 개발 환경 셋업
- 핵심 모듈 코드 리딩

## Week 2 - 실전 및 운영

- 기능 개발 실습
- 디버깅 & 트러블슈팅
- 배포 프로세스
- 운영 모니터링 & 대응
