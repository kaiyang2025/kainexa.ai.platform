"""
tests/integration/test_api_integration.py
API 통합 테스트
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import json
from datetime import datetime

from src.api.main import app
from src.core.config import settings


class TestAPIIntegration:
    """API 통합 테스트"""
    
    @pytest.fixture
    async def async_client(self):
        """비동기 HTTP 클라이언트"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def test_client(self):
        """동기 테스트 클라이언트"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """인증 헤더"""
        return {
            "Authorization": "Bearer test-jwt-token",
            "X-API-Key": "test-api-key"
        }
    
    @pytest.mark.asyncio
    async def test_health_check(self, async_client):
        """헬스체크 엔드포인트 테스트"""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_workflow_upload(self, async_client, auth_headers):
        """워크플로우 업로드 API 테스트"""
        workflow_dsl = {
            "metadata": {
                "namespace": "test",
                "name": "test_workflow",
                "version": "1.0.0"
            },
            "graph": {
                "nodes": [
                    {"id": "start", "type": "intent"},
                    {"id": "end", "type": "llm"}
                ],
                "edges": [
                    {"source": "start", "target": "end"}
                ]
            }
        }
        
        response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(workflow_dsl)},
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "workflow_id" in data
        assert data["status"] == "uploaded"
        assert data["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_workflow_compile(self, async_client, auth_headers):
        """워크플로우 컴파일 API 테스트"""
        # 먼저 업로드
        upload_response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": "..."},
            headers=auth_headers
        )
        workflow_id = upload_response.json()["workflow_id"]
        
        # 컴파일
        response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/compile",
            json={"version": "1.0.0"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "compiled"
        assert "compiled_graph" in data
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, async_client, auth_headers):
        """워크플로우 실행 API 테스트"""
        response = await async_client.post(
            "/api/v1/workflow/test/test_workflow/execute",
            json={
                "input": {
                    "message": "안녕하세요",
                    "user_id": "test_user"
                },
                "context": {
                    "session_id": "session_123"
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 202]
        data = response.json()
        assert "execution_id" in data
        assert "status" in data
    
    @pytest.mark.asyncio
    async def test_execution_status(self, async_client, auth_headers):
        """실행 상태 조회 API 테스트"""
        # 실행 시작
        exec_response = await async_client.post(
            "/api/v1/workflow/test/test_workflow/execute",
            json={"input": {"message": "test"}},
            headers=auth_headers
        )
        execution_id = exec_response.json()["execution_id"]
        
        # 상태 조회
        response = await async_client.get(
            f"/api/v1/executions/{execution_id}/status",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["execution_id"] == execution_id
        assert "status" in data
        assert "started_at" in data
    
    @pytest.mark.asyncio
    async def test_workflow_list(self, async_client, auth_headers):
        """워크플로우 목록 조회 API 테스트"""
        response = await async_client.get(
            "/api/v1/workflows",
            params={
                "namespace": "test",
                "page": 1,
                "size": 10
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data
        assert "pagination" in data
        assert isinstance(data["workflows"], list)
    
    @pytest.mark.asyncio
    async def test_workflow_versions(self, async_client, auth_headers):
        """워크플로우 버전 조회 API 테스트"""
        workflow_id = "test-workflow-id"
        
        response = await async_client.get(
            f"/api/v1/workflows/{workflow_id}/versions",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "versions" in data
        assert isinstance(data["versions"], list)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client):
        """Rate Limiting 테스트"""
        # 연속 요청
        responses = []
        for _ in range(20):
            response = await async_client.get("/health")
            responses.append(response)
        
        # Rate limit 헤더 확인
        last_response = responses[-1]
        assert "X-RateLimit-Limit" in last_response.headers
        assert "X-RateLimit-Remaining" in last_response.headers
        
        # 한계 초과 시 429 응답
        if int(last_response.headers["X-RateLimit-Remaining"]) == 0:
            response = await async_client.get("/health")
            assert response.status_code == 429
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """CORS 헤더 테스트"""
        response = await async_client.options(
            "/api/v1/workflows",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_client, auth_headers):
        """에러 처리 테스트"""
        # 잘못된 워크플로우 ID
        response = await async_client.get(
            "/api/v1/workflows/invalid-id",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "message" in data
        
        # 잘못된 요청 본문
        response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"invalid": "data"},
            headers=auth_headers
        )
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestWebSocketAPI:
    """WebSocket API 테스트"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, test_client):
        """WebSocket 연결 테스트"""
        with test_client.websocket_connect("/ws/chat") as websocket:
            # 초기 메시지
            websocket.send_json({
                "type": "init",
                "workflow": "chat_workflow",
                "user_id": "test_user"
            })
            
            response = websocket.receive_json()
            assert response["type"] == "init_success"
            assert "session_id" in response
    
    @pytest.mark.asyncio
    async def test_websocket_chat_flow(self, test_client):
        """WebSocket 채팅 플로우 테스트"""
        with test_client.websocket_connect("/ws/chat") as websocket:
            # 초기화
            websocket.send_json({
                "type": "init",
                "workflow": "chat_workflow"
            })
            websocket.receive_json()
            
            # 메시지 전송
            websocket.send_json({
                "type": "message",
                "content": "안녕하세요"
            })
            
            # 응답 수신
            response = websocket.receive_json()
            assert response["type"] == "response"
            assert "content" in response
            assert "metadata" in response
    
    @pytest.mark.asyncio
    async def test_websocket_streaming(self, test_client):
        """WebSocket 스트리밍 테스트"""
        with test_client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({
                "type": "stream_start",
                "prompt": "긴 응답을 생성해주세요"
            })
            
            # 스트리밍 청크 수신
            chunks = []
            while True:
                response = websocket.receive_json()
                if response["type"] == "stream_end":
                    break
                assert response["type"] == "stream_chunk"
                chunks.append(response["content"])
            
            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0


class TestAuthentication:
    """인증 테스트"""
    
    @pytest.mark.asyncio
    async def test_jwt_authentication(self, async_client):
        """JWT 인증 테스트"""
        # 로그인
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "username": "test_user",
                "password": "test_password"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        
        # 토큰으로 인증된 요청
        token = data["access_token"]
        auth_response = await async_client.get(
            "/api/v1/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert auth_response.status_code == 200
        user_data = auth_response.json()
        assert user_data["username"] == "test_user"
    
    @pytest.mark.asyncio
    async def test_api_key_authentication(self, async_client):
        """API 키 인증 테스트"""
        response = await async_client.get(
            "/api/v1/workflows",
            headers={"X-API-Key": "valid-api-key"}
        )
        
        assert response.status_code == 200
        
        # 잘못된 API 키
        response = await async_client.get(
            "/api/v1/workflows",
            headers={"X-API-Key": "invalid-api-key"}
        )
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_permission_check(self, async_client):
        """권한 체크 테스트"""
        # 일반 사용자 토큰
        user_token = "user-jwt-token"
        
        # 관리자 권한 필요한 엔드포인트
        response = await async_client.delete(
            "/api/v1/workflows/test-workflow",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        assert response.status_code == 403
        data = response.json()
        assert "permission denied" in data["message"].lower()


class TestMetrics:
    """메트릭 API 테스트"""
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics(self, async_client):
        """Prometheus 메트릭 엔드포인트 테스트"""
        response = await async_client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain"
        
        metrics_text = response.text
        assert "http_requests_total" in metrics_text
        assert "http_request_duration_seconds" in metrics_text
        assert "workflow_executions_total" in metrics_text
    
    @pytest.mark.asyncio
    async def test_custom_metrics(self, async_client, auth_headers):
        """커스텀 메트릭 조회 테스트"""
        response = await async_client.get(
            "/api/v1/metrics/workflows",
            params={
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "granularity": "daily"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "period" in data
        
        metrics = data["metrics"]
        assert "total_executions" in metrics
        assert "success_rate" in metrics
        assert "avg_latency" in metrics
        assert "total_tokens" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])