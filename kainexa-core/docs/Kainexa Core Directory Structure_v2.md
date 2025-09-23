# 🧹 Kainexa Core Directory Structure

```
kainexa-core/
│
├── 📋 README.md
├── 🔧 Makefile
├── 📦 pyproject.toml
├── 📝 requirements.txt
├── 📝 requirements-gpu.txt        # dev 의존성은 pyproject.toml로 통합
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml          # 모든 compose 파일 통합
├── ⚙️ .env.example
│
├── 📂 src/
│   ├── 📄 __init__.py
│   │
│   ├── 📂 api/                    # API 레이어 (간소화)
│   │   ├── 📄 main.py
│   │   ├── 📂 routes/
│   │   │   ├── 📄 health.py       # 헬스체크
│   │   │   ├── 📄 auth.py         # 인증
│   │   │   └── 📄 chat.py         # 통합 채팅 API ✅
│   │   ├── 📂 middleware/
│   │   │   └── 📄 middleware.py   # 모든 미들웨어 통합 ✅
│   │   └── 📂 schemas/
│   │       └── 📄 schemas.py      # 모든 스키마 통합 ✅
│   │
│   ├── 📂 core/                   # 핵심 설정 (간소화)
│   │   ├── 📄 config.py           # 모든 설정 통합 ✅
│   │   ├── 📄 database.py
│   │   ├── 📄 cache.py            # Redis 통합 ✅
│   │   └── 📄 exceptions.py
│   │
│   ├── 📂 orchestration/          # 오케스트레이션 (간소화)
│   │   ├── 📄 dsl_parser.py
│   │   ├── 📄 graph_executor.py
│   │   ├── 📄 policy_engine.py
│   │   ├── 📄 step_executors.py   # 모든 executor 통합 ✅
│   │   └── 📂 workflows/          # YAML 워크플로우만
│   │       └── 📄 default.yaml
│   │
│   ├── 📂 monitoring/             # 모니터링
│   │   ├── 📄 metrics.py          # collector + cost_tracker 통합 ✅
│   │   ├── 📄 gpu_monitor.py
│   │   └── 📄 tracer.py
│   │
│   ├── 📂 governance/             # RAG 거버넌스
│   │   ├── 📄 rag_pipeline.py
│   │   ├── 📄 document_processor.py
│   │   └── 📄 vector_store.py     # Qdrant 관리 통합 ✅
│   │
│   ├── 📂 auth/                   # MCP 권한
│   │   ├── 📄 mcp_permissions.py
│   │   └── 📄 jwt_manager.py      # JWT + 권한 통합 ✅
│   │
│   ├── 📂 models/                 # AI 모델 (간소화)
│   │   ├── 📄 tensor_parallel.py
│   │   ├── 📄 model_factory.py    # 모든 모델 로더 통합 ✅
│   │   └── 📄 inference.py        # 추론 실행 통합 ✅
│   │
│   ├── 📂 agents/                 # 에이전트 (핵심만)
│   │   ├── 📄 base_agent.py
│   │   ├── 📄 chat_agent.py       # 대화 통합 ✅
│   │   └── 📄 task_agent.py       # 작업 실행 ✅
│   │
│   ├── 📂 nlp/                    # NLP (간소화)
│   │   ├── 📄 korean_nlp.py       # 한국어 처리 통합 ✅
│   │   └── 📄 intent_classifier.py
│   │
│   └── 📂 utils/                  # 유틸리티
│       ├── 📄 logger.py
│       └── 📄 helpers.py          # validators + formatters 통합 ✅
│
├── 📂 configs/                    # 설정 (최소화)
│   └── 📂 workflows/              # 워크플로우만 유지
│       └── 📄 examples.yaml
│
├── 📂 tests/                      # 테스트 (간소화)
│   ├── 📄 conftest.py
│   ├── 📄 test_integration.py    # 통합 테스트
│   └── 📂 unit/                  # 핵심 단위 테스트만
│       ├── 📄 test_orchestration.py
│       ├── 📄 test_governance.py
│       └── 📄 test_auth.py
│
├── 📂 scripts/                    # 스크립트 (필수만)
│   ├── 📄 setup.sh               # 설정 통합 ✅
│   ├── 📄 run.sh                 # 실행 통합 ✅
│   └── 📄 gpu_setup.sh
│
├── 📂 docker/                     # Docker (간소화)
│   └── 📂 configs/               # 모든 설정 통합
│       ├── 📄 nginx.conf
│       ├── 📄 postgres.sql
│       └── 📄 prometheus.yml
│
└── 📂 monitoring/                 # 모니터링 설정
    └── 📂 dashboards/
        └── 📄 grafana.json
```
### 🎯 통합된 파일들

1. **API 통합**
```python
# schemas.py - 모든 Pydantic 스키마
class ConversationSchema(BaseModel): ...
class AgentSchema(BaseModel): ...
class AuthSchema(BaseModel): ...
```

2. **Executor 통합**
```python
# step_executors.py - 모든 실행자
class IntentExecutor(BaseExecutor): ...
class RetrievalExecutor(BaseExecutor): ...
class GenerationExecutor(BaseExecutor): ...
class ActionExecutor(BaseExecutor): ...
```

3. **설정 통합**
```python
# config.py - 모든 설정 중앙화
class Settings(BaseSettings):
    # API 설정
    api_prefix: str = "/api/v1"
    
    # 데이터베이스
    database_url: str
    redis_url: str
    
    # 모델 설정
    model_path: str
    tensor_parallel_size: int
    
    # 모니터링
    prometheus_port: int
    
    class Config:
        env_file = ".env"
```

4. **모델 팩토리**
```python
# model_factory.py - 모든 모델 로더
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: dict):
        if model_type == "solar":
            return SolarLLM(config)
        elif model_type == "openai":
            return OpenAIAdapter(config)
        # ... 필요시 추가
```

## ✅ 장점

1. **유지보수 용이**: 파일 수 50% 감소
2. **중복 제거**: 코드 재사용성 향상
3. **명확한 구조**: 3단계 이하 디렉토리
4. **빠른 개발**: 찾기 쉬운 파일 구조
5. **테스트 용이**: 통합된 모듈

## 🚀 실행 단순화

```bash
# Before: 복잡한 명령어
python src/models/run_tensor_parallel.py --config configs/models/solar_config.json

# After: 단순한 명령어
./scripts/run.sh --mode gpu
```

이렇게 정리하면 **실제 필요한 코드만** 남기고 **개발/운영 효율성**이 크게 향상됩니다!