# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is a **documentation-only repository** for handover of the Assetization system to the Vietnam GDC team (2-week program). It contains no application code — only markdown documents, quizzes, and lab materials.

## Repository Structure

- `docs/system-overview/` — System architecture diagrams, service catalog with source paths
- `docs/architecture/` — API endpoint maps, database schemas, infrastructure configs
- `docs/api-reference/` — (placeholder for detailed API docs)
- `labs/week1/` — Python quizzes and exercises (trilingual: Korean/English/Vietnamese)
- `labs/week2/` — (placeholder for hands-on labs)
- `resources/` — (placeholder for reference materials)

## What This Documents

9 microservices comprising the Assetization system (~56,000 lines):

| Service | Stack | Location |
|---------|-------|----------|
| assetization_mobile | React 19 + TS + Vite | `/Users/ryu/assetization_mobile` |
| assetization_authsso | Java + Spring Boot | `/Users/ryu/assetization_authsso` |
| assetization_auth | Python + FastAPI | `/Users/ryu/assetization_auth` |
| assetization_orchestrator | Python + FastAPI + LangGraph | `/Users/ryu/assetization_orchestrator` |
| assetization_mcp | Python + FastAPI + MCP | `/Users/ryu/assetization_mcp` |
| assetization_datacenter | Python + FastAPI | `/Users/ryu/assetization_datacenter` |
| assetization_api | Python + FastAPI + LangChain | `/Users/ryu/assetization_api` |
| keylook-officeplus | Python + FastAPI + FlagEmbedding | `/Users/ryu/keylook-officeplus` |
| keylook_script | Shell | `/Users/ryu/keylook_script` |

## Content Guidelines

- All handover documents and quizzes should be **trilingual**: 한국어 / English / Tiếng Việt, separated by `/`
- Quiz files use 4-choice multiple choice format with answer tables at the bottom
- Detailed explanation files pair with each quiz (`*-explained.md`)
- Reference the actual source repos above when writing about code paths or architecture
- Key doc entry point: `docs/system-overview/00-system-map.md` for the full architecture picture

## Workflow Rules

- **작업 완료 후 항상 커밋**: 모든 작업(파일 수정/생성)이 끝나면 반드시 git commit을 생성할 것
