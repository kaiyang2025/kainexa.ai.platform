# tests/integration/test_integration_full.py
"""
Kainexa Platform - 전체 시스템 통합 테스트
End-to-End 워크플로우 실행, 정책 적용, RAG 통합, 모델 라우팅 테스트
"""
import pytest
import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import yaml
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# 테스트 대상 모듈 임포트
from src.core.registry.workflow_manager import (
    WorkflowManager, WorkflowDSL, WorkflowStatus, Environment
)
from src.orchestration.graph_executor import GraphExecutor, GraphConfig
from src.orchestration.execution_context import ExecutionContext
from src.orchestration.step_executors import (
    IntentExecutor, LLMExecutor, APIExecutor, 
    ConditionExecutor, LoopExecutor
)
from src.orchestration.policy_engine import (
    PolicyEngine, PolicyAction, PolicyDecision,
    RoutingStrategy, SearchStrategy
)
from src.orchestration.model_router import ModelRouter, RoutingRequest
from src.governance.rag_pipeline import (
    RAGPipeline, DocumentType, ChunkingStrategy
)
from src.api.routes.workflow_routes import router as workflow_router
from fastapi.testclient import TestClient
from src.api.main import app

# ========== Test Fixtures ==========
@pytest.fixture(scope="module")
def event_loop():
    """이벤트 루프 fixture"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """테스트 데이터베이스 연결"""
    # 테스트 DB 설정 (실제 구현에서는 test DB 사용)
    import asyncpg
    conn = await asyncpg.connect(
        host=os.getenv("TEST_DB_HOST", "localhost"),
        port=os.getenv("TEST_DB_PORT", 5432),
        user=os.getenv("TEST_DB_USER", "test"),
        password=os.getenv("TEST_DB_PASSWORD", "test"),
        database=os.getenv("TEST_DB_NAME", "kainexa_test")
    )
    
    # 테스트 스키마 초기화
    await conn.execute("""
        DROP SCHEMA IF EXISTS test CASCADE;
        CREATE SCHEMA test;
        SET search_path TO test;
    """)
    
    # 테스트 테이블 생성 (workflow_schema.sql 내용)
    with open("database/schemas/workflow_schema.sql", "r") as f:
        schema_sql = f.read()
    await conn.execute(schema_sql)
    
    yield conn
    
    # 정리
    await conn.execute("DROP SCHEMA test CASCADE")
    await conn.close()

@pytest.fixture
def api_client():
    """FastAPI 테스트 클라이언트"""
    return TestClient(app)

@pytest.fixture
async def workflow_manager(test_db):
    """워크플로우 매니저 인스턴스"""
    manager = WorkflowManager()
    manager.db = test_db
    await manager.initialize()
    return manager

@pytest.fixture
def policy_engine():
    """정책 엔진 인스턴스"""
    config_path = "configs/policies/default_policies.yaml"
    engine = PolicyEngine()
    asyncio.create_task(engine.initialize_policies(config_path))
    return engine

@pytest.fixture
def model_router():
    """모델 라우터 인스턴스"""
    return ModelRouter({
        'summarizer_model': 'slm-ko-3b',
        'max_retries': 3
    })

@pytest.fixture
async def rag_pipeline():
    """RAG 파이프라인 인스턴스"""
    pipeline = RAGPipeline({
        'qdrant_host': 'localhost',
        'qdrant_port': 6333,
        'collection_name': 'test_collection'
    })
    return pipeline

@pytest.fixture
def sample_workflow_dsl():
    """샘플 워크플로우 DSL"""
    return {
        "version": "1.0.0",
        "workflow": {
            "name": "customer-service-flow",
            "description": "고객 서비스 통합 워크플로우",
            "author": "test@kainexa.ai",
            "tags": ["customer", "test"]
        },
        "nodes": [
            {
                "id": "intent_node",
                "type": "intent",
                "config": {
                    "model": "solar-intent",
                    "confidence_threshold": 0.7
                }
            },
            {
                "id": "rag_node",
                "type": "knowledge",
                "config": {
                    "strategy": "hybrid",
                    "top_k": 5
                }
            },
            {
                "id": "llm_node",
                "type": "llm",
                "config": {
                    "model": "solar-10.7b",
                    "max_tokens": 512,
                    "temperature": 0.7
                }
            },
            {
                "id": "condition_node",
                "type": "condition",
                "config": {
                    "condition": "confidence > 0.8"
                }
            },
            {
                "id": "api_node",
                "type": "api",
                "config": {
                    "endpoint": "https://api.example.com/action",
                    "method": "POST"
                }
            }
        ],
        "edges": [
            {"from": "start", "to": "intent_node"},
            {"from": "intent_node", "to": "rag_node"},
            {"from": "rag_node", "to": "llm_node"},
            {"from": "llm_node", "to": "condition_node"},
            {"from": "condition_node", "to": "api_node", "condition": "true"},
            {"from": "condition_node", "to": "end", "condition": "false"},
            {"from": "api_node", "to": "end"}
        ],
        "policies": {
            "sla": {
                "max_latency_seconds": 5
            },
            "cost_limit": {
                "max_per_session": 0.1
            },
            "retry": {
                "max_attempts": 2
            }
        }
    }

@pytest.fixture
async def sample_documents(tmp_path):
    """테스트용 샘플 문서 생성"""
    docs = []
    
    # 텍스트 문서
    text_file = tmp_path / "manual.txt"
    text_file.write_text("""
    환불 정책: 구매 후 14일 이내에 환불이 가능합니다.
    교환 정책: 제품 하자 시 30일 이내 교환 가능합니다.
    고객 서비스: 평일 9시-18시 운영합니다.
    """)
    docs.append((str(text_file), DocumentType.TEXT))
    
    # JSON 문서
    json_file = tmp_path / "faq.json"
    json_file.write_text(json.dumps({
        "faqs": [
            {"q": "배송 기간은?", "a": "평균 2-3일 소요됩니다"},
            {"q": "결제 방법은?", "a": "카드, 계좌이체 가능합니다"}
        ]
    }, ensure_ascii=False))
    docs.append((str(json_file), DocumentType.JSON))
    
    return docs

# ========== Integration Test Cases ==========

class TestWorkflowIntegration:
    """워크플로우 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_lifecycle(self, 
                                              workflow_manager,
                                              sample_workflow_dsl,
                                              api_client):
        """
        완전한 워크플로우 생명주기 테스트
        1. 생성 → 2. 컴파일 → 3. 시뮬레이션 → 4. 퍼블리시 → 5. 실행
        """
        
        # 1. 워크플로우 생성
        workflow = await workflow_manager.create_workflow(
            namespace="test",
            name="integration-test",
            dsl=sample_workflow_dsl,
            created_by="test@kainexa.ai"
        )
        assert workflow.id is not None
        assert workflow.status == WorkflowStatus.UPLOADED
        
        # 2. 컴파일
        compiled = await workflow_manager.compile_workflow(
            workflow.id,
            workflow.version
        )
        assert compiled.status == WorkflowStatus.COMPILED
        assert compiled.compiled_graph is not None
        
        # 3. 시뮬레이션
        simulation_result = await workflow_manager.simulate_workflow(
            workflow.id,
            workflow.version,
            input_data={"text": "환불하고 싶습니다"},
            context={"session_id": "test-sim"}
        )
        assert simulation_result["status"] == "success"
        assert "execution_path" in simulation_result
        
        # 4. 퍼블리시
        published = await workflow_manager.publish_workflow(
            workflow.id,
            workflow.version,
            environment=Environment.DEV
        )
        assert published["status"] == "published"
        assert published["environment"] == "dev"
        
        # 5. API를 통한 실행
        response = api_client.post(
            f"/api/v1/workflow/test/integration-test/execute",
            json={
                "session_id": "test-exec",
                "input": {"text": "환불 요청합니다"},
                "context": {"language": "ko-KR"}
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["execution_id"] is not None
        assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_workflow_versioning(self, workflow_manager):
        """워크플로우 버전 관리 테스트"""
        
        # v1.0.0 생성
        v1 = await workflow_manager.create_workflow(
            namespace="test",
            name="versioned",
            dsl={"version": "1.0.0", "workflow": {}},
            created_by="test@kainexa.ai"
        )
        
        # v1.0.1 업데이트
        v1_update = await workflow_manager.update_workflow(
            v1.id,
            dsl={"version": "1.0.1", "workflow": {"updated": True}},
            updated_by="test@kainexa.ai"
        )
        assert v1_update.version == "1.0.1"
        
        # v2.0.0 메이저 업데이트
        v2 = await workflow_manager.create_version(
            v1.id,
            version="2.0.0",
            dsl={"version": "2.0.0", "workflow": {"major": True}},
            created_by="test@kainexa.ai"
        )
        assert v2.version == "2.0.0"
        
        # 버전 히스토리 조회
        history = await workflow_manager.get_version_history(v1.id)
        assert len(history) == 3
        assert history[0]["version"] == "1.0.0"
        assert history[2]["version"] == "2.0.0"
    
    @pytest.mark.asyncio
    async def test_workflow_rollback(self, workflow_manager):
        """워크플로우 롤백 테스트"""
        
        # 초기 버전 생성 및 퍼블리시
        workflow = await workflow_manager.create_workflow(
            namespace="test",
            name="rollback-test",
            dsl={"version": "1.0.0", "workflow": {"stable": True}},
            created_by="test@kainexa.ai"
        )
        
        await workflow_manager.publish_workflow(
            workflow.id,
            "1.0.0",
            Environment.PROD
        )
        
        # 새 버전 생성 및 퍼블리시
        await workflow_manager.create_version(
            workflow.id,
            version="1.1.0",
            dsl={"version": "1.1.0", "workflow": {"buggy": True}},
            created_by="test@kainexa.ai"
        )
        
        await workflow_manager.publish_workflow(
            workflow.id,
            "1.1.0",
            Environment.PROD
        )
        
        # 이전 버전으로 롤백
        rollback_result = await workflow_manager.activate_version(
            workflow.id,
            "1.0.0",
            Environment.PROD
        )
        
        assert rollback_result["active_version"] == "1.0.0"
        assert rollback_result["previous_version"] == "1.1.0"


class TestPolicyEngineIntegration:
    """정책 엔진 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_policy_chain_execution(self, policy_engine):
        """정책 체인 실행 테스트"""
        
        # 복합 정책 시나리오 설정
        step = Mock()
        step.name = "test_step"
        step.type = "llm"
        step.params = {"model": "gpt-4", "max_tokens": 1000}
        step.rate_limit = {
            "max_requests": 5,
            "period_seconds": 60,
            "scope": "session"
        }
        step.confidence_threshold = 0.8
        
        context = {
            "session_id": "policy-test",
            "start_time": time.time(),
            "confidence": 0.6,  # 낮은 신뢰도
            "sentiment": {"label": "negative", "score": 0.9},  # 부정적 감정
            "user_message": "긴급! 당장 처리해주세요!"  # 긴급 키워드
        }
        
        global_policies = {
            "sla": {"max_latency_seconds": 30},
            "cost_limit": {"max_per_session": 1.0}
        }
        
        # 정책 평가
        decision = await policy_engine.evaluate(step, context, global_policies)
        
        # 우선순위에 따라 에스컬레이션이 트리거되어야 함
        assert decision.action == PolicyAction.ESCALATE
        assert "escalation_target" in decision.metadata
    
    @pytest.mark.asyncio
    async def test_cost_tracking_and_fallback(self, policy_engine):
        """비용 추적 및 폴백 테스트"""
        
        session_id = "cost-test"
        
        # 비용 누적
        for i in range(5):
            policy_engine.cost_tracker.add_cost(session_id, 0.25)
        
        # 비용 한도 초과 확인
        step = Mock()
        step.type = "llm"
        step.params = {"model": "gpt-4", "max_tokens": 1000}
        
        context = {"session_id": session_id, "start_time": time.time()}
        global_policies = {
            "cost_limit": {
                "max_per_session": 1.0,
                "fallback_model": "slm-ko-3b"
            }
        }
        
        decision = await policy_engine.evaluate(step, context, global_policies)
        
        assert decision.action == PolicyAction.FALLBACK
        assert decision.metadata["fallback_model"] == "slm-ko-3b"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_across_sessions(self, policy_engine):
        """세션 간 Rate Limiting 테스트"""
        
        step = Mock()
        step.name = "api_call"
        step.rate_limit = {
            "max_requests": 10,
            "period_seconds": 60,
            "scope": "global"  # 전역 제한
        }
        
        # 여러 세션에서 동시 요청
        tasks = []
        for i in range(15):
            context = {
                "session_id": f"session-{i}",
                "start_time": time.time()
            }
            tasks.append(policy_engine.evaluate(step, context))
        
        results = await asyncio.gather(*tasks)
        
        # 처음 10개는 통과, 나머지는 차단
        continue_count = sum(1 for r in results if r.action == PolicyAction.CONTINUE)
        throttle_count = sum(1 for r in results if r.action == PolicyAction.THROTTLE)
        
        assert continue_count == 10
        assert throttle_count == 5


class TestModelRouterIntegration:
    """모델 라우터 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_adaptive_routing(self, model_router):
        """적응형 라우팅 테스트"""
        
        # 초기 요청들로 히스토리 구축
        for i in range(5):
            request = RoutingRequest(
                text="기술 문서 작성 요청",
                max_tokens=256,
                strategy=RoutingStrategy.BALANCED,
                metadata={"session_id": f"session-{i}"}
            )
            
            decision = await model_router.route(request)
            
            # 성공 메트릭 기록
            model_router.metrics[decision.primary_model]["success"] += 1
            model_router.metrics[decision.primary_model]["total_latency_ms"] += 100
        
        # 적응형 라우팅 요청
        adaptive_request = RoutingRequest(
            text="기술 문서 작성 요청",  # 유사한 요청
            max_tokens=256,
            strategy=RoutingStrategy.ADAPTIVE,
            metadata={"session_id": "adaptive-test"}
        )
        
        decision = await model_router.route(adaptive_request)
        
        assert decision.strategy_used == RoutingStrategy.ADAPTIVE
        assert "similar requests" in decision.reasoning
    
    @pytest.mark.asyncio
    async def test_summarization_pipeline(self, model_router):
        """요약 경유 파이프라인 테스트"""
        
        # 긴 텍스트
        long_text = " ".join(["이것은 매우 긴 텍스트입니다."] * 500)
        
        with patch.object(model_router, "_execute_model", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {
                "text": "요약된 텍스트",
                "tokens_used": 100,
                "latency_ms": 500
            }
            
            result = await model_router.execute_with_summarization(
                "solar-10.7b",
                long_text,
                max_tokens=512,
                summarize_threshold=1000
            )
            
            assert result["summarization_used"] == True
            assert "original_length" in result
            assert "summary_length" in result
            # 요약 모델과 주 모델 둘 다 호출되었는지 확인
            assert mock_exec.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_gpu_resource_management(self, model_router):
        """GPU 리소스 관리 테스트"""
        
        # GPU 상태 확인
        initial_status = model_router.gpu_manager.get_status()
        available_gpus = initial_status["available_gpus"]
        
        # 여러 모델 할당
        allocations = []
        for i in range(3):
            request = RoutingRequest(
                text=f"Request {i}",
                max_tokens=256,
                strategy=RoutingStrategy.QUALITY_FIRST
            )
            decision = await model_router.route(request)
            
            if decision.gpu_allocation:
                allocations.append((decision.primary_model, decision.gpu_allocation))
        
        # GPU 사용률 확인
        current_status = model_router.gpu_manager.get_status()
        
        for gpu_info in current_status["gpus"]:
            assert gpu_info["utilization_percent"] >= 0
            assert len(gpu_info["allocated_models"]) >= 0
        
        # 할당 해제
        for model_name, allocation in allocations:
            if model_name in model_router.models:
                await model_router.gpu_manager.deallocate(
                    model_name,
                    model_router.models[model_name]
                )
        
        # 해제 후 상태 확인
        final_status = model_router.gpu_manager.get_status()
        assert final_status["available_gpus"] >= available_gpus


class TestRAGPipelineIntegration:
    """RAG 파이프라인 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_document_ingestion_and_retrieval(self, 
                                                   rag_pipeline,
                                                   sample_documents):
        """문서 수집 및 검색 테스트"""
        
        # 문서 수집
        doc_ids = []
        for file_path, doc_type in sample_documents:
            doc_id = await rag_pipeline.ingest_document(
                file_path,
                doc_type,
                ChunkingStrategy.FIXED_SIZE,
                metadata={"test": True}
            )
            doc_ids.append(doc_id)
        
        assert len(doc_ids) == 2
        
        # 검색 테스트
        search_results = await rag_pipeline.retrieve(
            query="환불 정책",
            top_k=3,
            strategy=SearchStrategy.SIMILARITY
        )
        
        assert len(search_results) > 0
        assert search_results[0].score >= 0.5
        
        # 하이브리드 검색
        hybrid_results = await rag_pipeline.retrieve(
            query="배송 기간",
            top_k=3,
            strategy=SearchStrategy.HYBRID,
            rerank=True
        )
        
        assert len(hybrid_results) > 0
        if hybrid_results[0].rerank_score:
            assert hybrid_results[0].rerank_score != hybrid_results[0].score
    
    @pytest.mark.asyncio
    async def test_rag_context_generation(self, rag_pipeline, sample_documents):
        """RAG 컨텍스트 생성 테스트"""
        
        # 문서 인덱싱
        for file_path, doc_type in sample_documents:
            await rag_pipeline.ingest_document(file_path, doc_type)
        
        # 전체 RAG 파이프라인 실행
        rag_context = await rag_pipeline.process_query(
            query="고객 서비스 운영 시간은?",
            top_k=5,
            strategy=SearchStrategy.SIMILARITY,
            rerank=True,
            template_name="korean",
            max_context_length=1000
        )
        
        assert rag_context.query == "고객 서비스 운영 시간은?"
        assert len(rag_context.retrieved_chunks) > 0
        assert len(rag_context.enhanced_prompt) > len(rag_context.query)
        assert rag_context.processing_time_ms > 0
        
        # 프롬프트에 컨텍스트가 포함되었는지 확인
        assert "[참고 정보]" in rag_context.enhanced_prompt
        assert "[질문]" in rag_context.enhanced_prompt
    
    @pytest.mark.asyncio
    async def test_batch_document_processing(self, rag_pipeline, tmp_path):
        """배치 문서 처리 테스트"""
        
        # 여러 문서 생성
        documents = []
        for i in range(10):
            file_path = tmp_path / f"doc_{i}.txt"
            file_path.write_text(f"Document {i} content")
            documents.append((str(file_path), DocumentType.TEXT, {"batch": i}))
        
        # 배치 수집
        results = await rag_pipeline.batch_ingest(
            documents,
            ChunkingStrategy.FIXED_SIZE,
            batch_size=3
        )
        
        assert len(results) == 10
        assert all(r is not None for r in results)
        
        # 메트릭 확인
        metrics = rag_pipeline.get_metrics()
        assert metrics["ingestion"]["total_documents"] >= 10


class TestEndToEndScenarios:
    """End-to-End 시나리오 테스트"""
    
    @pytest.mark.asyncio
    async def test_customer_service_scenario(self,
                                            workflow_manager,
                                            policy_engine,
                                            model_router,
                                            rag_pipeline,
                                            sample_workflow_dsl,
                                            sample_documents):
        """
        고객 서비스 시나리오 전체 테스트
        1. 문서 인덱싱
        2. 워크플로우 생성 및 퍼블리시
        3. 고객 질문 처리
        4. 정책 적용 및 라우팅
        5. 응답 생성
        """
        
        # 1. 고객 서비스 문서 인덱싱
        for file_path, doc_type in sample_documents:
            await rag_pipeline.ingest_document(
                file_path,
                doc_type,
                metadata={"category": "customer_service"}
            )
        
        # 2. 워크플로우 생성 및 퍼블리시
        workflow = await workflow_manager.create_workflow(
            namespace="cs",
            name="customer-service",
            dsl=sample_workflow_dsl,
            created_by="admin@kainexa.ai"
        )
        
        await workflow_manager.compile_workflow(workflow.id, workflow.version)
        await workflow_manager.publish_workflow(
            workflow.id,
            workflow.version,
            Environment.PROD
        )
        
        # 3. Graph Executor 설정
        graph_config = GraphConfig.from_dsl(sample_workflow_dsl)
        
        # Step Executors 초기화
        step_executors = {
            "intent": IntentExecutor(),
            "llm": LLMExecutor(model_router),
            "api": APIExecutor(),
            "condition": ConditionExecutor(),
            "loop": LoopExecutor()
        }
        
        # Metrics Collector Mock
        metrics_collector = Mock()
        metrics_collector.record_step = AsyncMock()
        
        graph_executor = GraphExecutor(
            policy_engine=policy_engine,
            metrics_collector=metrics_collector,
            step_executors=step_executors
        )
        
        # 4. 고객 질문 처리
        test_cases = [
            {
                "input": "환불하고 싶습니다",
                "expected_intent": "refund",
                "expected_policy": PolicyAction.CONTINUE
            },
            {
                "input": "긴급! 즉시 처리 필요!",
                "expected_intent": "urgent",
                "expected_policy": PolicyAction.ESCALATE
            },
            {
                "input": "제품이 고장났어요",
                "expected_intent": "defect",
                "expected_policy": PolicyAction.CONTINUE
            }
        ]
        
        for test_case in test_cases:
            # 실행 컨텍스트 생성
            context = ExecutionContext(
                session_id=f"test-{test_case['expected_intent']}",
                input_data={"text": test_case["input"]},
                variables={},
                metadata={"language": "ko-KR"}
            )
            
            # Mock 응답 설정
            with patch.object(graph_executor, "execute_step") as mock_exec:
                mock_exec.return_value = AsyncMock(return_value={
                    "output": {
                        "intent": test_case["expected_intent"],
                        "confidence": 0.85,
                        "response": "처리되었습니다"
                    },
                    "next_step": "end"
                })
                
                # 그래프 실행
                result = await graph_executor.execute_graph(
                    graph_config,
                    context
                )
                
                assert result.status == "completed"
                assert result.output is not None
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self,
                                     workflow_manager,
                                     policy_engine,
                                     model_router):
        """
        고부하 시나리오 테스트
        - 동시 다중 세션
        - Rate limiting
        - 리소스 경쟁
        """
        
        # 간단한 워크플로우 생성
        simple_workflow = {
            "version": "1.0.0",
            "workflow": {"name": "load-test"},
            "nodes": [
                {"id": "llm", "type": "llm", "config": {"model": "slm-ko-3b"}}
            ],
            "edges": [
                {"from": "start", "to": "llm"},
                {"from": "llm", "to": "end"}
            ]
        }
        
        workflow = await workflow_manager.create_workflow(
            namespace="test",
            name="load-test",
            dsl=simple_workflow,
            created_by="test@kainexa.ai"
        )
        
        # 동시 요청 생성
        async def process_request(session_id: str):
            request = RoutingRequest(
                text=f"Request from {session_id}",
                max_tokens=128,
                strategy=RoutingStrategy.LATENCY_OPTIMIZED,
                metadata={"session_id": session_id}
            )
            
            try:
                decision = await model_router.route(request)
                return {"session_id": session_id, "model": decision.primary_model, "status": "success"}
            except Exception as e:
                return {"session_id": session_id, "error": str(e), "status": "failed"}
        
        # 100개 동시 요청
        tasks = [process_request(f"session-{i}") for i in range(100)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # 성능 검증
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        
        assert success_count > 80  # 80% 이상 성공
        assert duration < 30  # 30초 이내 완료
        
        print(f"Load test: {success_count}/100 succeeded in {duration:.2f}s")
        
        # Rate limiting 확인
        throttled = [r for r in results if "error" in r and "rate limit" in r.get("error", "").lower()]
        print(f"Rate limited: {len(throttled)} requests")
    
    @pytest.mark.asyncio
    async def test_failure_recovery_scenario(self,
                                            workflow_manager,
                                            policy_engine,
                                            model_router):
        """
        장애 복구 시나리오 테스트
        - 모델 실패 시 폴백
        - 재시도 로직
        - 에러 핸들링
        """
        
        # 실패를 시뮬레이션할 Mock 설정
        call_count = 0
        
        async def mock_execute_model(model_name, prompt, max_tokens, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                # 처음 2번은 실패
                raise Exception(f"Model {model_name} failed")
            else:
                # 3번째는 성공
                return {
                    "text": "Recovered response",
                    "tokens_used": 100,
                    "model": model_name
                }
        
        with patch.object(model_router, "_execute_model", side_effect=mock_execute_model):
            result = await model_router.execute_with_retry(
                "solar-10.7b",
                "Test prompt",
                max_tokens=256
            )
            
            assert result["text"] == "Recovered response"
            assert call_count == 3  # 2번 실패 + 1번 성공


class TestMonitoringAndMetrics:
    """모니터링 및 메트릭 테스트"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self,
                                     policy_engine,
                                     model_router,
                                     rag_pipeline):
        """메트릭 수집 테스트"""
        
        # 여러 작업 실행
        for i in range(10):
            # 정책 평가
            step = Mock()
            step.name = f"step_{i}"
            context = {"session_id": f"metric-test-{i}", "start_time": time.time()}
            
            await policy_engine.evaluate(step, context)
            
            # 모델 라우팅
            request = RoutingRequest(
                text=f"Request {i}",
                max_tokens=128,
                strategy=RoutingStrategy.BALANCED
            )
            await model_router.route(request)
        
        # 메트릭 수집
        policy_stats = policy_engine.get_stats()
        router_metrics = model_router.get_metrics()
        rag_metrics = rag_pipeline.get_metrics()
        
        # 검증
        assert policy_stats["total_decisions"] >= 10
        assert router_metrics["total_requests"] >= 10
        
        # 메트릭 포맷 확인
        assert "by_step" in policy_stats
        assert "models" in router_metrics
        
        print("=== Collected Metrics ===")
        print(f"Policy decisions: {policy_stats['total_decisions']}")
        print(f"Routing requests: {router_metrics['total_requests']}")
        print(f"RAG metrics: {json.dumps(rag_metrics, indent=2)}")


# ========== Performance Tests ==========

class TestPerformance:
    """성능 테스트"""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_performance(self, workflow_manager):
        """워크플로우 실행 성능 테스트"""
        
        # 성능 측정용 워크플로우
        perf_workflow = {
            "version": "1.0.0",
            "workflow": {"name": "perf-test"},
            "nodes": [
                {"id": f"node_{i}", "type": "condition", "config": {}}
                for i in range(10)
            ],
            "edges": [
                {"from": f"node_{i}", "to": f"node_{i+1}"}
                for i in range(9)
            ]
        }
        perf_workflow["edges"].insert(0, {"from": "start", "to": "node_0"})
        perf_workflow["edges"].append({"from": "node_9", "to": "end"})
        
        workflow = await workflow_manager.create_workflow(
            namespace="perf",
            name="performance",
            dsl=perf_workflow,
            created_by="test@kainexa.ai"
        )
        
        # 컴파일 성능
        compile_start = time.time()
        await workflow_manager.compile_workflow(workflow.id, workflow.version)
        compile_time = time.time() - compile_start
        
        assert compile_time < 1.0  # 1초 이내
        
        # 시뮬레이션 성능
        sim_start = time.time()
        await workflow_manager.simulate_workflow(
            workflow.id,
            workflow.version,
            input_data={"test": True}
        )
        sim_time = time.time() - sim_start
        
        assert sim_time < 2.0  # 2초 이내
        
        print(f"Performance: Compile={compile_time:.3f}s, Simulate={sim_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_rag_search_performance(self, rag_pipeline, tmp_path):
        """RAG 검색 성능 테스트"""
        
        # 대량 문서 생성
        for i in range(100):
            file_path = tmp_path / f"perf_doc_{i}.txt"
            file_path.write_text(f"Performance test document {i} " * 100)
            
            await rag_pipeline.ingest_document(
                str(file_path),
                DocumentType.TEXT,
                ChunkingStrategy.FIXED_SIZE
            )
        
        # 검색 성능 측정
        search_times = []
        
        for i in range(10):
            start = time.time()
            results = await rag_pipeline.retrieve(
                query=f"Performance test query {i}",
                top_k=10,
                strategy=SearchStrategy.SIMILARITY
            )
            search_time = time.time() - start
            search_times.append(search_time)
        
        avg_search_time = sum(search_times) / len(search_times)
        p95_search_time = sorted(search_times)[int(len(search_times) * 0.95)]
        
        print(f"RAG Search: Avg={avg_search_time:.3f}s, P95={p95_search_time:.3f}s")
        
        assert avg_search_time < 0.5  # 평균 500ms 이내
        assert p95_search_time < 1.0  # P95 1초 이내


# ========== Test Report Generator ==========

class TestReportGenerator:
    """테스트 리포트 생성"""
    
    @pytest.fixture(autouse=True)
    def setup_report(self):
        """리포트 설정"""
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }
        yield
        self.generate_report()
    
    def add_test_result(self, name: str, status: str, duration: float, details: Dict = None):
        """테스트 결과 추가"""
        self.report["tests"].append({
            "name": name,
            "status": status,
            "duration": duration,
            "details": details or {}
        })
    
    def generate_report(self):
        """리포트 생성"""
        total = len(self.report["tests"])
        passed = sum(1 for t in self.report["tests"] if t["status"] == "passed")
        failed = total - passed
        
        self.report["summary"] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "total_duration": sum(t["duration"] for t in self.report["tests"])
        }
        
        # 리포트 저장
        report_path = Path("test_reports") / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2)
        
        print(f"\n=== Test Report ===")
        print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
        print(f"Pass Rate: {self.report['summary']['pass_rate']:.1f}%")
        print(f"Report saved to: {report_path}")


# ========== Main Test Runner ==========

if __name__ == "__main__":
    # pytest 실행 옵션
    pytest.main([
        __file__,
        "-v",  # verbose
        "-s",  # stdout 출력
        "--tb=short",  # 간단한 traceback
        "--cov=src",  # 커버리지
        "--cov-report=html",  # HTML 리포트
        "--cov-report=term-missing",  # 터미널 리포트
        "--html=test_reports/integration_test.html",  # HTML 리포트
        "--self-contained-html",  # 독립형 HTML
        "--junit-xml=test_reports/integration_test.xml",  # JUnit XML
        "--maxfail=5",  # 5개 실패 시 중단
        "--durations=10",  # 가장 느린 10개 테스트 표시
        "-m", "not slow",  # slow 마크 제외
    ])