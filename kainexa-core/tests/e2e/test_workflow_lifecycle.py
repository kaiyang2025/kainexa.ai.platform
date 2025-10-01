"""
tests/e2e/test_workflow_lifecycle.py
워크플로우 생명주기 End-to-End 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import json
from httpx import AsyncClient


class TestWorkflowLifecycle:
    """워크플로우 생명주기 E2E 테스트"""
    
    @pytest.fixture
    def sample_workflow(self):
        """샘플 워크플로우 DSL"""
        return {
            "metadata": {
                "namespace": "test_lifecycle",
                "name": "lifecycle_workflow",
                "version": "1.0.0",
                "description": "Lifecycle test workflow"
            },
            "graph": {
                "nodes": [
                    {
                        "id": "start",
                        "type": "intent",
                        "config": {
                            "patterns": ["start", "begin"],
                            "confidence_threshold": 0.7
                        }
                    },
                    {
                        "id": "process",
                        "type": "llm",
                        "config": {
                            "model": "gpt-3.5-turbo",
                            "prompt": "Process the request"
                        }
                    },
                    {
                        "id": "end",
                        "type": "api",
                        "config": {
                            "url": "https://api.example.com/complete",
                            "method": "POST"
                        }
                    }
                ],
                "edges": [
                    {"source": "start", "target": "process"},
                    {"source": "process", "target": "end"}
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
    async def test_complete_lifecycle(self, async_client, auth_headers, sample_workflow):
        """완전한 워크플로우 생명주기 테스트"""
        
        # 1. Upload
        upload_response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(sample_workflow)},
            headers=auth_headers
        )
        
        assert upload_response.status_code == 201
        upload_data = upload_response.json()
        workflow_id = upload_data["workflow_id"]
        assert upload_data["status"] == "uploaded"
        
        # 2. Validate
        validate_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/validate",
            headers=auth_headers
        )
        
        assert validate_response.status_code == 200
        validate_data = validate_response.json()
        assert validate_data["is_valid"] == True
        
        # 3. Compile
        compile_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/compile",
            json={"version": "1.0.0"},
            headers=auth_headers
        )
        
        assert compile_response.status_code == 200
        compile_data = compile_response.json()
        assert compile_data["status"] == "compiled"
        
        # 4. Simulate
        simulate_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/simulate",
            json={
                "version": "1.0.0",
                "input_data": {"message": "test simulation"}
            },
            headers=auth_headers
        )
        
        assert simulate_response.status_code == 200
        simulate_data = simulate_response.json()
        assert simulate_data["status"] == "completed"
        
        # 5. Publish to Staging
        staging_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/publish",
            json={
                "version": "1.0.0",
                "environment": "staging"
            },
            headers=auth_headers
        )
        
        assert staging_response.status_code == 200
        staging_data = staging_response.json()
        assert staging_data["environment"] == "staging"
        
        # 6. Test in Staging
        staging_test_response = await async_client.post(
            "/api/v1/workflow/test_lifecycle/lifecycle_workflow/execute",
            json={"input": {"message": "staging test"}},
            headers={**auth_headers, "X-Environment": "staging"}
        )
        
        assert staging_test_response.status_code == 200
        
        # 7. Publish to Production
        prod_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/publish",
            json={
                "version": "1.0.0",
                "environment": "production"
            },
            headers=auth_headers
        )
        
        assert prod_response.status_code == 200
        prod_data = prod_response.json()
        assert prod_data["environment"] == "production"
        
        # 8. Monitor Executions
        await asyncio.sleep(1)  # 일부 실행 시뮬레이션
        
        metrics_response = await async_client.get(
            f"/api/v1/workflows/{workflow_id}/metrics",
            params={"environment": "production"},
            headers=auth_headers
        )
        
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        assert "total_executions" in metrics_data
        
        # 9. Deactivate
        deactivate_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/deactivate",
            json={"environment": "production"},
            headers=auth_headers
        )
        
        assert deactivate_response.status_code == 200
        deactivate_data = deactivate_response.json()
        assert deactivate_data["status"] == "deactivated"
    
    @pytest.mark.asyncio
    async def test_version_management(self, async_client, auth_headers, sample_workflow):
        """버전 관리 테스트"""
        
        # 초기 버전 업로드
        versions = ["1.0.0", "1.0.1", "1.1.0", "2.0.0"]
        workflow_id = None
        
        for version in versions:
            workflow = sample_workflow.copy()
            workflow["metadata"]["version"] = version
            
            response = await async_client.post(
                "/api/v1/workflows/upload",
                json={"dsl_content": json.dumps(workflow)},
                headers=auth_headers
            )
            
            if workflow_id is None:
                workflow_id = response.json()["workflow_id"]
            
            # 각 버전 컴파일
            await async_client.post(
                f"/api/v1/workflows/{workflow_id}/compile",
                json={"version": version},
                headers=auth_headers
            )
        
        # 버전 목록 조회
        versions_response = await async_client.get(
            f"/api/v1/workflows/{workflow_id}/versions",
            headers=auth_headers
        )
        
        versions_data = versions_response.json()
        assert len(versions_data["versions"]) == 4
        assert versions_data["versions"][0]["version"] == "2.0.0"  # 최신 버전
        
        # 특정 버전 활성화
        activate_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/activate",
            json={
                "version": "1.1.0",
                "environment": "production"
            },
            headers=auth_headers
        )
        
        assert activate_response.status_code == 200
        assert activate_response.json()["active_version"] == "1.1.0"
    
    @pytest.mark.asyncio
    async def test_rollback_scenario(self, async_client, auth_headers, sample_workflow):
        """롤백 시나리오 테스트"""
        
        # 안정된 버전 배포
        workflow = sample_workflow.copy()
        workflow["metadata"]["version"] = "1.0.0"
        
        upload_response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(workflow)},
            headers=auth_headers
        )
        workflow_id = upload_response.json()["workflow_id"]
        
        # 컴파일 및 배포
        await async_client.post(
            f"/api/v1/workflows/{workflow_id}/compile",
            json={"version": "1.0.0"},
            headers=auth_headers
        )
        
        await async_client.post(
            f"/api/v1/workflows/{workflow_id}/publish",
            json={"version": "1.0.0", "environment": "production"},
            headers=auth_headers
        )
        
        # 새 버전 배포 (문제가 있는 버전 시뮬레이션)
        workflow["metadata"]["version"] = "2.0.0"
        workflow["graph"]["nodes"].append({
            "id": "problematic_node",
            "type": "api",
            "config": {"url": "https://broken.api.com"}
        })
        
        await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(workflow)},
            headers=auth_headers
        )
        
        await async_client.post(
            f"/api/v1/workflows/{workflow_id}/compile",
            json={"version": "2.0.0"},
            headers=auth_headers
        )
        
        await async_client.post(
            f"/api/v1/workflows/{workflow_id}/publish",
            json={"version": "2.0.0", "environment": "production"},
            headers=auth_headers
        )
        
        # 에러 감지 시뮬레이션
        await asyncio.sleep(2)
        
        # 롤백 실행
        rollback_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/rollback",
            json={
                "target_version": "1.0.0",
                "environment": "production",
                "reason": "High error rate detected"
            },
            headers=auth_headers
        )
        
        assert rollback_response.status_code == 200
        rollback_data = rollback_response.json()
        assert rollback_data["rolled_back_to"] == "1.0.0"
        assert rollback_data["success"] == True
        
        # 현재 활성 버전 확인
        status_response = await async_client.get(
            f"/api/v1/workflows/{workflow_id}/status",
            params={"environment": "production"},
            headers=auth_headers
        )
        
        assert status_response.json()["active_version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_ab_testing(self, async_client, auth_headers, sample_workflow):
        """A/B 테스팅 테스트"""
        
        # 버전 A 준비
        workflow_a = sample_workflow.copy()
        workflow_a["metadata"]["version"] = "1.0.0"
        
        upload_a = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(workflow_a)},
            headers=auth_headers
        )
        workflow_id = upload_a.json()["workflow_id"]
        
        # 버전 B 준비 (다른 모델 사용)
        workflow_b = sample_workflow.copy()
        workflow_b["metadata"]["version"] = "2.0.0"
        workflow_b["graph"]["nodes"][1]["config"]["model"] = "gpt-4"
        
        await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(workflow_b)},
            headers=auth_headers
        )
        
        # 두 버전 모두 컴파일
        for version in ["1.0.0", "2.0.0"]:
            await async_client.post(
                f"/api/v1/workflows/{workflow_id}/compile",
                json={"version": version},
                headers=auth_headers
            )
        
        # A/B 테스트 설정
        ab_test_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/ab-test",
            json={
                "test_name": "model_comparison",
                "version_a": "1.0.0",
                "version_b": "2.0.0",
                "traffic_split": 0.5,
                "duration_hours": 24,
                "environment": "production"
            },
            headers=auth_headers
        )
        
        assert ab_test_response.status_code == 200
        ab_test_data = ab_test_response.json()
        test_id = ab_test_data["test_id"]
        
        # 트래픽 시뮬레이션
        executions = []
        for i in range(100):
            exec_response = await async_client.post(
                f"/api/v1/workflow/{workflow_id}/execute",
                json={"input": {"message": f"test {i}"}},
                headers=auth_headers
            )
            executions.append(exec_response.json())
        
        # A/B 테스트 결과 조회
        results_response = await async_client.get(
            f"/api/v1/ab-tests/{test_id}/results",
            headers=auth_headers
        )
        
        results_data = results_response.json()
        assert "version_a_metrics" in results_data
        assert "version_b_metrics" in results_data
        assert "statistical_significance" in results_data
        
        # 승자 선택
        winner_response = await async_client.post(
            f"/api/v1/ab-tests/{test_id}/conclude",
            json={"winner": "version_b"},
            headers=auth_headers
        )
        
        assert winner_response.status_code == 200
        assert winner_response.json()["deployed_version"] == "2.0.0"
    
    @pytest.mark.asyncio
    async def test_canary_deployment(self, async_client, auth_headers, sample_workflow):
        """카나리 배포 테스트"""
        
        # 현재 버전
        current_workflow = sample_workflow.copy()
        current_workflow["metadata"]["version"] = "1.0.0"
        
        upload_response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(current_workflow)},
            headers=auth_headers
        )
        workflow_id = upload_response.json()["workflow_id"]
        
        # 새 버전
        new_workflow = sample_workflow.copy()
        new_workflow["metadata"]["version"] = "2.0.0"
        
        await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(new_workflow)},
            headers=auth_headers
        )
        
        # 카나리 배포 시작 (10% 트래픽)
        canary_response = await async_client.post(
            f"/api/v1/workflows/{workflow_id}/canary",
            json={
                "new_version": "2.0.0",
                "initial_traffic": 0.1,
                "increment": 0.2,
                "interval_minutes": 30,
                "environment": "production"
            },
            headers=auth_headers
        )
        
        assert canary_response.status_code == 200
        canary_data = canary_response.json()
        canary_id = canary_data["canary_id"]
        
        # 카나리 상태 확인
        status_response = await async_client.get(
            f"/api/v1/canary/{canary_id}/status",
            headers=auth_headers
        )
        
        status_data = status_response.json()
        assert status_data["current_traffic"] == 0.1
        assert status_data["status"] == "in_progress"
        
        # 메트릭 확인 후 진행
        proceed_response = await async_client.post(
            f"/api/v1/canary/{canary_id}/proceed",
            headers=auth_headers
        )
        
        assert proceed_response.status_code == 200
        assert proceed_response.json()["new_traffic"] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])