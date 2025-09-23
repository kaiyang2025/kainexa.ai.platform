# Kainexa Core 재설계_v2

## 환경 정보
- **OS**: Ubuntu 24.04.3 LTS (GNU/Linux 6.8.0-79-generic x86_64)
- **GPU**: NVIDIA RTX 3090 ×2 (24GB ×2)
- **배포 형태**: On-Premise (Docker Compose / Kubernetes Helm Chart)

---

## A. 아키텍처 4대 표준화 축

### 1. 오케스트레이션 (실행그래프·정책)
- DSL 기반 실행그래프: YAML/JSON으로 대화·액션·외부 API 호출을 순차/병렬 구성
- 정책 레이어:  
  - 룰 기반 정책 (Fallback, Escalation)  
  - 확률 기반 정책 (LLM confidence score 기반)  
  - 비용·성능 최적화 정책 (Latency SLA, GPU 사용량 제어)

**DSL 예시 (YAML):**
```yaml
graph:
  - step: intent_classify
  - step: retrieve_knowledge
    policy: if confidence < 0.7 then escalate
  - step: llm_generate
  - step: response_postprocess
```

---

### 2. 관측성 / 비용 관리
- **Observability**
  - Prometheus + Grafana: 응답시간, GPU 메모리, 토큰 사용량 추적
  - OpenTelemetry 기반 추적
- **비용 제어**
  - GPU Time·Token 단위 모니터링
  - 세션별 비용 태깅
  - SLA/월별 한도 설정 후 초과 시 fallback (경량모델)

---

### 3. RAG 거버넌스
- 데이터 파이프라인: 문서 → Chunking → Embedding → VectorDB(Qdrant) 저장
- 메타데이터 정책: 문서 출처, 최신성, 접근권한 태깅
- 품질 관리: Re-ranking (Cross-Encoder), 답변 시 출처 자동 표기

**OpenAPI 예시 (Knowledge API):**
```yaml
paths:
  /v1/knowledge/documents:
    post:
      summary: Upload new document
      responses:
        "200":
          description: Document ingested
```

---

### 4. MCP 권한 모델
- **권한 레벨**
  - User: 대화 요청
  - Agent: API 호출·Action 실행
  - Admin: 데이터/정책 관리
- **정책 스키마 예시**
```json
{
  "role": "agent",
  "permissions": ["retrieve:knowledge", "invoke:action:payment"],
  "limits": {"tokens_per_min": 2000}
}
```

---

## B. GPU 활용 전략 (On-Prem, RTX3090 ×2, 24GB×2)

### B-1. 모델/런타임 배치 (안 2 — Tensor Parallel 2-way)
- 대형 LLM (Solar-10.7B): 텐서 병렬 (2-way)로 두 GPU 분산 실행
- 경량 LLM (Fallback): 단일 GPU (3090 #2)에서 실행
- RAG/벡터검색(Qdrant): CPU + RAM, GPU 비사용
- 추가 최적화: DeepSpeed ZeRO-2, FlashAttention, INT8 quantization, KV-Cache

**배치 다이어그램**
```
GPU#0 ─┬─ LLM Shard 1 (Tensor Parallel)
       │
GPU#1 ─┴─ LLM Shard 2 (Tensor Parallel)
       └─ Lightweight LLM / Inference
```

---

## C. 개발 일정 (On-Prem Ubuntu 배포 기준)

| Sprint (주차) | 주요 개발 내용 | Deliverables |
|---------------|---------------|--------------|
| 1-4 (Week 1-4) | 기초 인프라 | Docker Compose, API Gateway, Redis, Postgres |
| 5-8 (Week 5-8) | 대화 엔진 핵심 | Solar LLM 텐서 병렬 통합, 기본 대화 API |
| 9-12 (Week 9-12) | 한국어 NLP | 의도 분류, 개체 추출, 존댓말 변환기 |
| 13-16 (Week 13-16) | RAG KB 구축 | 문서 파이프라인, 벡터 DB, 거버넌스 |
| 17-20 (Week 17-20) | MCP 권한 & 모니터링 | 권한 정책, Prometheus/Grafana, 비용 추적 |
| 21-24 (Week 21-24) | 통합 테스트 & 최적화 | SLA 성능검증, GPU 효율화, On-Prem 패키징 |

---

## D. Ubuntu On-Premise 납품 형태
- **배포 패키지**
  - Docker Compose (Core API, NLP, RAG, MCP, Monitoring)
  - Helm Chart (옵션, Kubernetes 배포 지원)
- **운영 지원**
  - GPU 2×RTX3090 텐서 병렬 설정 포함
  - 설치 매뉴얼 (Ubuntu 24.04.3 기준)
  - 초기 성능 튜닝 스크립트 제공
