"""
tests/e2e/test_production_scenarios.py
프로덕션 시나리오 End-to-End 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import json
import random
from httpx import AsyncClient
from concurrent.futures import ThreadPoolExecutor


class TestProductionScenarios:
    """프로덕션 시나리오 E2E 테스트"""
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self, async_client, auth_headers, setup_workflow):
        """고부하 시나리오 테스트"""
        
        async def execute_request(session_num):
            """단일 요청 실행"""
            try:
                response = await async_client.post(
                    "/api/v1/workflow/production/chat/execute",
                    json={
                        "input": {
                            "message": f"Test message {session_num}",
                            "user_id": f"user_{session_num}"
                        }
                    },
                    headers=auth_headers,
                    timeout=10.0
                )
                return {
                    "session": session_num,
                    "status": response.status_code,
                    "latency": response.elapsed.total_seconds() * 1000
                }
            except Exception as e:
                return {
                    "session": session_num,
                    "status": "error",
                    "error": str(e)
                }
        
        # 동시 요청 생성
        concurrent_requests = 100
        tasks = [execute_request(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        
        # 결과 분석
        successful = [r for r in results if isinstance(r.get("status"), int) and r["status"] == 200]
        failed = [r for r in results if r.get("status") != 200]
        
        success_rate = len(successful) / len(results)
        assert success_rate >= 0.95, f"Success rate {success_rate} is below 95%"
        
        if successful:
            avg_latency = sum(r["latency"] for r in successful) / len(successful)
            p95_latency = sorted([r["latency"] for r in successful])[int(len(successful) * 0.95)]
            
            assert avg_latency < 3000, f"Average latency {avg_latency}ms exceeds 3000ms"
            assert p95_latency < 5000, f"P95 latency {p95_latency}ms exceeds 5000ms"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, async_client, auth_headers):
        """서킷 브레이커 테스트"""
        
        # 의도적으로 실패하는 워크플로우 설정
        failing_workflow = {
            "metadata": {
                "namespace": "test",
                "name": "failing_workflow"
            },
            "graph": {
                "nodes": [{
                    "id": "fail_node",
                    "type": "api",
                    "config": {
                        "url": "https://nonexistent.api.com/fail"
                    }
                }]
            }
        }
        
        # 워크플로우 배포
        upload_response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(failing_workflow)},
            headers=auth_headers
        )
        
        # 연속 실패 트리거
        failure_count = 0
        circuit_open = False
        
        for i in range(20):
            response = await async_client.post(
                "/api/v1/workflow/test/failing_workflow/execute",
                json={"input": {"message": f"test {i}"}},
                headers=auth_headers
            )
            
            if response.status_code == 503:  # Circuit breaker opened
                circuit_open = True
                break
            elif response.status_code != 200:
                failure_count += 1
        
        assert circuit_open, "Circuit breaker did not open after multiple failures"
        
        # 서킷 브레이커 상태 확인
        status_response = await async_client.get(
            "/api/v1/circuit-breaker/status",
            params={"service": "failing_workflow"},
            headers=auth_headers
        )
        
        assert status_response.json()["state"] == "open"
        
        # 일정 시간 후 half-open 상태 확인
        await asyncio.sleep(5)
        
        status_response = await async_client.get(
            "/api/v1/circuit-breaker/status",
            params={"service": "failing_workflow"},
            headers=auth_headers
        )
        
        assert status_response.json()["state"] in ["half_open", "closed"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting_per_user(self, async_client, auth_headers):
        """사용자별 Rate Limiting 테스트"""
        
        user_id = "rate_limit_test_user"
        
        # Rate limit 도달까지 요청
        responses = []
        for i in range(150):  # Rate limit: 100 requests per minute
            response = await async_client.post(
                "/api/v1/workflow/test/chat/execute",
                json={
                    "input": {
                        "message": f"Message {i}",
                        "user_id": user_id
                    }
                },
                headers=auth_headers
            )
            responses.append(response)
            
            if response.status_code == 429:  # Too Many Requests
                break
        
        # Rate limit 도달 확인
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Rate limiting not enforced"
        
        # Rate limit 헤더 확인
        last_response = responses[-1]
        assert "X-RateLimit-Limit" in last_response.headers
        assert "X-RateLimit-Remaining" in last_response.headers
        assert "X-RateLimit-Reset" in last_response.headers
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, async_client, auth_headers):
        """Graceful Degradation 테스트"""
        
        # 복잡한 워크플로우 실행 (일부 서비스 실패 시뮬레이션)
        response = await async_client.post(
            "/api/v1/workflow/complex/multi_service/execute",
            json={
                "input": {
                    "message": "Get product recommendations with reviews",
                    "user_id": "test_user"
                },
                "context": {
                    "simulate_failures": ["review_service"]
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # 일부 기능은 실패했지만 핵심 기능은 작동
        assert result["status"] == "partial_success"
        assert "recommendations" in result["data"]  # 핵심 기능
        assert result["data"].get("reviews") is None  # 실패한 기능
        assert "degraded_services" in result["metadata"]
        assert "review_service" in result["metadata"]["degraded_services"]
    
    @pytest.mark.asyncio
    async def test_data_consistency(self, async_client, auth_headers):
        """데이터 일관성 테스트"""
        
        workflow_id = "consistency_test"
        
        # 동시에 같은 워크플로우 업데이트 시도
        async def update_workflow(version):
            return await async_client.post(
                f"/api/v1/workflows/{workflow_id}/update",
                json={
                    "version": version,
                    "changes": {"description": f"Updated by {version}"}
                },
                headers=auth_headers
            )
        
        # 동시 업데이트 실행
        tasks = [update_workflow(f"v{i}") for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 하나만 성공해야 함 (optimistic locking)
        successful = [r for r in results if not isinstance(r, Exception) and r.status_code == 200]
        assert len(successful) == 1, "Multiple concurrent updates succeeded"
        
        # 최종 상태 확인
        final_state = await async_client.get(
            f"/api/v1/workflows/{workflow_id}",
            headers=auth_headers
        )
        
        assert final_state.status_code == 200
        assert "version" in final_state.json()
    
    @pytest.mark.asyncio
    async def test_disaster_recovery(self, async_client, auth_headers):
        """재해 복구 시나리오 테스트"""
        
        # 1. 현재 상태 백업
        backup_response = await async_client.post(
            "/api/v1/admin/backup",
            json={"include": ["workflows", "configurations", "policies"]},
            headers=auth_headers
        )
        
        assert backup_response.status_code == 200
        backup_id = backup_response.json()["backup_id"]
        
        # 2. 일부 데이터 변경/삭제 시뮬레이션
        await async_client.delete(
            "/api/v1/workflows/test_workflow",
            headers=auth_headers
        )
        
        # 3. 백업에서 복구
        restore_response = await async_client.post(
            "/api/v1/admin/restore",
            json={"backup_id": backup_id},
            headers=auth_headers
        )
        
        assert restore_response.status_code == 200
        assert restore_response.json()["status"] == "restored"
        
        # 4. 복구된 데이터 확인
        check_response = await async_client.get(
            "/api/v1/workflows/test_workflow",
            headers=auth_headers
        )
        
        assert check_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting(self, async_client, auth_headers):
        """모니터링 및 알림 테스트"""
        
        # 알림 규칙 설정
        alert_rule = {
            "name": "high_error_rate",
            "condition": {
                "metric": "error_rate",
                "operator": ">",
                "threshold": 0.1,
                "window": "5m"
            },
            "actions": [
                {"type": "email", "to": "admin@example.com"},
                {"type": "slack", "channel": "#alerts"}
            ]
        }
        
        rule_response = await async_client.post(
            "/api/v1/monitoring/alerts",
            json=alert_rule,
            headers=auth_headers
        )
        
        rule_id = rule_response.json()["rule_id"]
        
        # 에러 생성하여 알림 트리거
        for i in range(20):
            await async_client.post(
                "/api/v1/workflow/test/error_workflow/execute",
                json={"input": {"trigger_error": True}},
                headers=auth_headers
            )
        
        # 알림 확인
        await asyncio.sleep(2)
        
        alerts_response = await async_client.get(
            "/api/v1/monitoring/alerts/triggered",
            params={"rule_id": rule_id},
            headers=auth_headers
        )
        
        alerts = alerts_response.json()["alerts"]
        assert len(alerts) > 0
        assert alerts[0]["rule_name"] == "high_error_rate"
    
    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, async_client, auth_headers):
        """멀티 테넌트 격리 테스트"""
        
        # 테넌트 A 설정
        tenant_a_headers = {**auth_headers, "X-Tenant-ID": "tenant_a"}
        workflow_a = await async_client.post(
            "/api/v1/workflows/upload",
            json={
                "dsl_content": json.dumps({
                    "metadata": {"namespace": "tenant_a", "name": "workflow_a"}
                })
            },
            headers=tenant_a_headers
        )
        
        # 테넌트 B 설정
        tenant_b_headers = {**auth_headers, "X-Tenant-ID": "tenant_b"}
        workflow_b = await async_client.post(
            "/api/v1/workflows/upload",
            json={
                "dsl_content": json.dumps({
                    "metadata": {"namespace": "tenant_b", "name": "workflow_b"}
                })
            },
            headers=tenant_b_headers
        )
        
        # 테넌트 A가 테넌트 B의 워크플로우에 접근 시도
        unauthorized_response = await async_client.get(
            f"/api/v1/workflows/{workflow_b.json()['workflow_id']}",
            headers=tenant_a_headers
        )
        
        assert unauthorized_response.status_code == 403
        
        # 각 테넌트는 자신의 워크플로우만 조회 가능
        tenant_a_list = await async_client.get(
            "/api/v1/workflows",
            headers=tenant_a_headers
        )
        
        workflows_a = tenant_a_list.json()["workflows"]
        assert all(w["namespace"] == "tenant_a" for w in workflows_a)
    
    @pytest.mark.asyncio
    async def test_compliance_audit_trail(self, async_client, auth_headers):
        """규정 준수 감사 추적 테스트"""
        
        # PII 데이터 포함 요청
        response = await async_client.post(
            "/api/v1/workflow/compliance/process/execute",
            json={
                "input": {
                    "message": "Process user John Doe, SSN: 123-45-6789",
                    "user_id": "test_user"
                }
            },
            headers=auth_headers
        )
        
        execution_id = response.json()["execution_id"]
        
        # 감사 로그 조회
        audit_response = await async_client.get(
            "/api/v1/audit/logs",
            params={
                "execution_id": execution_id,
                "include_pii": False  # PII는 마스킹
            },
            headers=auth_headers
        )
        
        audit_logs = audit_response.json()["logs"]
        assert len(audit_logs) > 0
        
        # PII 마스킹 확인
        for log in audit_logs:
            assert "123-45-6789" not in json.dumps(log)
            assert "XXX-XX-XXXX" in json.dumps(log) or "[MASKED]" in json.dumps(log)
        
        # 규정 준수 보고서 생성
        report_response = await async_client.post(
            "/api/v1/compliance/report",
            json={
                "period": "last_30_days",
                "include": ["pii_handling", "data_retention", "access_logs"]
            },
            headers=auth_headers
        )
        
        assert report_response.status_code == 200
        report = report_response.json()
        assert "pii_handling" in report
        assert report["pii_handling"]["masked_count"] > 0
    
    @pytest.mark.asyncio
    async def test_zero_downtime_deployment(self, async_client, auth_headers):
        """무중단 배포 테스트"""
        
        # 현재 버전으로 지속적인 요청 생성
        async def continuous_requests(stop_event):
            """지속적인 요청 생성"""
            results = []
            while not stop_event.is_set():
                try:
                    response = await async_client.post(
                        "/api/v1/workflow/production/service/execute",
                        json={"input": {"message": "continuous test"}},
                        headers=auth_headers,
                        timeout=5.0
                    )
                    results.append({
                        "time": datetime.now(),
                        "status": response.status_code,
                        "version": response.headers.get("X-Version")
                    })
                except Exception as e:
                    results.append({
                        "time": datetime.now(),
                        "status": "error",
                        "error": str(e)
                    })
                await asyncio.sleep(0.1)
            return results
        
        # 배포 시작
        stop_event = asyncio.Event()
        request_task = asyncio.create_task(continuous_requests(stop_event))
        
        # 새 버전 배포 (Blue-Green)
        deploy_response = await async_client.post(
            "/api/v1/deployment/blue-green",
            json={
                "new_version": "2.0.0",
                "health_check_url": "/health",
                "switch_strategy": "immediate"
            },
            headers=auth_headers
        )
        
        assert deploy_response.status_code == 200
        
        # 10초 동안 요청 계속
        await asyncio.sleep(10)
        stop_event.set()
        results = await request_task
        
        # 다운타임 확인
        errors = [r for r in results if r["status"] == "error"]
        error_rate = len(errors) / len(results) if results else 0
        
        assert error_rate < 0.01, f"Error rate {error_rate} exceeds 1%"
        
        # 버전 전환 확인
        versions = [r.get("version") for r in results if r.get("version")]
        assert "1.0.0" in versions
        assert "2.0.0" in versions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])