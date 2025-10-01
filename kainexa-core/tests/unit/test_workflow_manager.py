"""
tests/unit/test_workflow_manager.py
워크플로우 매니저 단위 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from src.core.registry.workflow_manager import WorkflowManager, WorkflowStatus
from src.api.schemas.workflow_schemas import (
    WorkflowUploadRequest,
    WorkflowCompileRequest,
    WorkflowSimulateRequest,
    WorkflowPublishRequest
)


class TestWorkflowManager:
    """워크플로우 매니저 테스트"""
    
    @pytest.fixture
    def workflow_manager(self):
        """워크플로우 매니저 인스턴스"""
        return WorkflowManager()
    
    @pytest.fixture
    def sample_dsl(self):
        """샘플 DSL"""
        return {
            "metadata": {
                "namespace": "customer_service",
                "name": "order_inquiry",
                "version": "1.0.0",
                "description": "주문 조회 워크플로우"
            },
            "graph": {
                "nodes": [
                    {
                        "id": "intent_1",
                        "type": "intent",
                        "config": {
                            "patterns": ["주문 조회", "배송 상태"],
                            "confidence_threshold": 0.8
                        }
                    },
                    {
                        "id": "llm_1",
                        "type": "llm",
                        "config": {
                            "model": "gpt-3.5-turbo",
                            "prompt": "주문 정보를 조회합니다."
                        }
                    }
                ],
                "edges": [
                    {
                        "source": "intent_1",
                        "target": "llm_1",
                        "condition": "confidence > 0.8"
                    }
                ]
            },
            "policies": {
                "sla": {
                    "max_latency_ms": 3000,
                    "timeout_ms": 5000
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_upload_workflow(self, workflow_manager, sample_dsl):
        """워크플로우 업로드 테스트"""
        # DSL 업로드
        request = WorkflowUploadRequest(
            namespace="customer_service",
            name="order_inquiry",
            dsl_content=json.dumps(sample_dsl)
        )
        
        with patch.object(workflow_manager, 'db', new=AsyncMock()) as mock_db:
            mock_db.execute.return_value = None
            mock_db.fetch_one.return_value = {"id": "test-workflow-id"}
            
            result = await workflow_manager.upload_workflow(request)
            
            assert result.workflow_id == "test-workflow-id"
            assert result.status == "uploaded"
            assert result.version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_compile_workflow(self, workflow_manager, sample_dsl):
        """워크플로우 컴파일 테스트"""
        request = WorkflowCompileRequest(
            workflow_id="test-workflow-id",
            version="1.0.0"
        )
        
        with patch.object(workflow_manager, 'db', new=AsyncMock()) as mock_db:
            mock_db.fetch_one.return_value = {
                "dsl_raw": json.dumps(sample_dsl),
                "status": "uploaded"
            }
            mock_db.execute.return_value = None
            
            result = await workflow_manager.compile_workflow(request)
            
            assert result.status == "compiled"
            assert result.compiled_graph is not None
            assert "nodes" in result.compiled_graph
            assert "edges" in result.compiled_graph
    
    @pytest.mark.asyncio
    async def test_validate_dsl(self, workflow_manager, sample_dsl):
        """DSL 유효성 검증 테스트"""
        # 정상 DSL
        is_valid, errors = workflow_manager.validate_dsl(sample_dsl)
        assert is_valid == True
        assert len(errors) == 0
        
        # 필수 필드 누락
        invalid_dsl = {
            "metadata": {
                "namespace": "test"
                # name 필드 누락
            }
        }
        is_valid, errors = workflow_manager.validate_dsl(invalid_dsl)
        assert is_valid == False
        assert len(errors) > 0
        
        # 잘못된 노드 타입
        invalid_dsl = sample_dsl.copy()
        invalid_dsl["graph"]["nodes"][0]["type"] = "invalid_type"
        is_valid, errors = workflow_manager.validate_dsl(invalid_dsl)
        assert is_valid == False
        assert any("invalid node type" in err.lower() for err in errors)
    
    @pytest.mark.asyncio
    async def test_simulate_workflow(self, workflow_manager, sample_dsl):
        """워크플로우 시뮬레이션 테스트"""
        request = WorkflowSimulateRequest(
            workflow_id="test-workflow-id",
            version="1.0.0",
            input_data={
                "message": "주문 조회하고 싶어요",
                "user_id": "test_user"
            }
        )
        
        with patch.object(workflow_manager, 'db', new=AsyncMock()) as mock_db:
            mock_db.fetch_one.return_value = {
                "compiled_graph": sample_dsl["graph"],
                "status": "compiled"
            }
            
            # GraphExecutor 모킹
            with patch('src.core.executor.graph_executor.GraphExecutor') as MockExecutor:
                mock_executor = MockExecutor.return_value
                mock_executor.execute = AsyncMock(return_value={
                    "status": "completed",
                    "outputs": {"response": "주문 정보입니다."},
                    "metrics": {"latency": 1500}
                })
                
                result = await workflow_manager.simulate_workflow(request)
                
                assert result.status == "completed"
                assert "response" in result.outputs
                assert result.metrics["latency"] == 1500
    
    @pytest.mark.asyncio
    async def test_publish_workflow(self, workflow_manager):
        """워크플로우 퍼블리시 테스트"""
        request = WorkflowPublishRequest(
            workflow_id="test-workflow-id",
            version="1.0.0",
            environment="production"
        )
        
        with patch.object(workflow_manager, 'db', new=AsyncMock()) as mock_db:
            # 워크플로우 상태 확인
            mock_db.fetch_one.return_value = {
                "status": "compiled",
                "version": "1.0.0"
            }
            mock_db.execute.return_value = None
            
            result = await workflow_manager.publish_workflow(request)
            
            assert result.status == "published"
            assert result.environment == "production"
            assert result.message == "Successfully published to production"
    
    @pytest.mark.asyncio
    async def test_rollback_workflow(self, workflow_manager):
        """워크플로우 롤백 테스트"""
        workflow_id = "test-workflow-id"
        target_version = "0.9.0"
        environment = "production"
        
        with patch.object(workflow_manager, 'db', new=AsyncMock()) as mock_db:
            # 이전 버전 존재 확인
            mock_db.fetch_one.return_value = {
                "version": target_version,
                "status": "published"
            }
            mock_db.execute.return_value = None
            
            result = await workflow_manager.rollback_workflow(
                workflow_id, target_version, environment
            )
            
            assert result.success == True
            assert result.rolled_back_to == target_version
    
    @pytest.mark.asyncio
    async def test_get_workflow_metrics(self, workflow_manager):
        """워크플로우 메트릭 조회 테스트"""
        workflow_id = "test-workflow-id"
        
        with patch.object(workflow_manager, 'db', new=AsyncMock()) as mock_db:
            mock_db.fetch_all.return_value = [
                {
                    "execution_id": "exec-1",
                    "latency_ms": 1500,
                    "status": "completed",
                    "tokens_in": 100,
                    "tokens_out": 150
                },
                {
                    "execution_id": "exec-2",
                    "latency_ms": 2000,
                    "status": "completed",
                    "tokens_in": 120,
                    "tokens_out": 180
                }
            ]
            
            metrics = await workflow_manager.get_workflow_metrics(workflow_id)
            
            assert metrics.total_executions == 2
            assert metrics.avg_latency == 1750
            assert metrics.success_rate == 1.0
            assert metrics.total_tokens == 550
    
    def test_check_cyclic_dependencies(self, workflow_manager, sample_dsl):
        """순환 종속성 체크 테스트"""
        # 정상 그래프 (순환 없음)
        has_cycle = workflow_manager.check_cyclic_dependencies(
            sample_dsl["graph"]
        )
        assert has_cycle == False
        
        # 순환 그래프
        cyclic_graph = {
            "nodes": [
                {"id": "node1", "type": "llm"},
                {"id": "node2", "type": "llm"},
                {"id": "node3", "type": "llm"}
            ],
            "edges": [
                {"source": "node1", "target": "node2"},
                {"source": "node2", "target": "node3"},
                {"source": "node3", "target": "node1"}  # 순환 생성
            ]
        }
        
        has_cycle = workflow_manager.check_cyclic_dependencies(cyclic_graph)
        assert has_cycle == True
    
    @pytest.mark.asyncio
    async def test_version_management(self, workflow_manager):
        """버전 관리 테스트"""
        workflow_id = "test-workflow-id"
        
        with patch.object(workflow_manager, 'db', new=AsyncMock()) as mock_db:
            mock_db.fetch_all.return_value = [
                {"version": "1.2.0", "created_at": datetime(2024, 1, 15)},
                {"version": "1.1.0", "created_at": datetime(2024, 1, 10)},
                {"version": "1.0.0", "created_at": datetime(2024, 1, 5)}
            ]
            
            versions = await workflow_manager.get_workflow_versions(workflow_id)
            
            assert len(versions) == 3
            assert versions[0]["version"] == "1.2.0"  # 최신 버전
            
            # 특정 버전 조회
            mock_db.fetch_one.return_value = {
                "version": "1.1.0",
                "dsl_raw": json.dumps(sample_dsl),
                "status": "published"
            }
            
            version_detail = await workflow_manager.get_workflow_version(
                workflow_id, "1.1.0"
            )
            
            assert version_detail["version"] == "1.1.0"
            assert version_detail["status"] == "published"


class TestWorkflowValidation:
    """워크플로우 유효성 검증 테스트"""
    
    def test_validate_node_configs(self):
        """노드 설정 유효성 검증"""
        from src.core.registry.workflow_validator import WorkflowValidator
        
        validator = WorkflowValidator()
        
        # LLM 노드 검증
        llm_node = {
            "id": "llm_1",
            "type": "llm",
            "config": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        is_valid, errors = validator.validate_node(llm_node)
        assert is_valid == True
        
        # 필수 설정 누락
        invalid_node = {
            "id": "llm_2",
            "type": "llm",
            "config": {
                # model 필드 누락
                "temperature": 0.7
            }
        }
        is_valid, errors = validator.validate_node(invalid_node)
        assert is_valid == False
        assert "model" in str(errors)
    
    def test_validate_edge_conditions(self):
        """엣지 조건 유효성 검증"""
        from src.core.registry.workflow_validator import WorkflowValidator
        
        validator = WorkflowValidator()
        
        # 유효한 조건
        valid_conditions = [
            "confidence > 0.8",
            "output.status == 'success'",
            "contains(text, 'order')",
            "true"
        ]
        
        for condition in valid_conditions:
            is_valid = validator.validate_condition(condition)
            assert is_valid == True
        
        # 잘못된 조건
        invalid_conditions = [
            "confidence >>> 0.8",  # 문법 오류
            "exec('malicious')",   # 위험한 함수
            ""                     # 빈 조건
        ]
        
        for condition in invalid_conditions:
            is_valid = validator.validate_condition(condition)
            assert is_valid == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])