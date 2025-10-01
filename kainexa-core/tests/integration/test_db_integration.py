"""
tests/integration/test_db_integration.py
데이터베이스 통합 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import json

from src.core.database import Base, get_db
from src.core.config import settings


class TestDatabaseIntegration:
    """데이터베이스 통합 테스트"""
    
    @pytest.fixture
    async def db_engine(self):
        """테스트용 DB 엔진"""
        engine = create_async_engine(
            settings.TEST_DATABASE_URL,
            echo=True
        )
        
        # 테이블 생성
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield engine
        
        # 테이블 삭제
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        await engine.dispose()
    
    @pytest.fixture
    async def db_session(self, db_engine):
        """테스트용 DB 세션"""
        async_session = sessionmaker(
            db_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with async_session() as session:
            yield session
    
    @pytest.mark.asyncio
    async def test_workflow_crud(self, db_session):
        """워크플로우 CRUD 테스트"""
        # CREATE
        workflow_data = {
            "namespace": "test",
            "name": "test_workflow",
            "description": "테스트 워크플로우",
            "created_by": "test_user"
        }
        
        result = await db_session.execute(
            """
            INSERT INTO workflows (namespace, name, description, created_by)
            VALUES ($1, $2, $3, $4)
            RETURNING id, created_at
            """,
            workflow_data["namespace"],
            workflow_data["name"],
            workflow_data["description"],
            workflow_data["created_by"]
        )
        
        workflow = result.fetchone()
        assert workflow is not None
        workflow_id = workflow["id"]
        
        # READ
        result = await db_session.execute(
            "SELECT * FROM workflows WHERE id = $1",
            workflow_id
        )
        
        fetched_workflow = result.fetchone()
        assert fetched_workflow["name"] == "test_workflow"
        assert fetched_workflow["namespace"] == "test"
        
        # UPDATE
        await db_session.execute(
            "UPDATE workflows SET description = $1 WHERE id = $2",
            "업데이트된 설명",
            workflow_id
        )
        await db_session.commit()
        
        result = await db_session.execute(
            "SELECT description FROM workflows WHERE id = $1",
            workflow_id
        )
        updated = result.fetchone()
        assert updated["description"] == "업데이트된 설명"
        
        # DELETE (Soft delete)
        await db_session.execute(
            "UPDATE workflows SET deleted_at = $1 WHERE id = $2",
            datetime.now(),
            workflow_id
        )
        await db_session.commit()
        
        result = await db_session.execute(
            "SELECT deleted_at FROM workflows WHERE id = $1",
            workflow_id
        )
        deleted = result.fetchone()
        assert deleted["deleted_at"] is not None
    
    @pytest.mark.asyncio
    async def test_workflow_version_management(self, db_session):
        """워크플로우 버전 관리 테스트"""
        # 워크플로우 생성
        workflow_id = await self._create_test_workflow(db_session)
        
        # 여러 버전 추가
        versions = ["1.0.0", "1.0.1", "1.1.0", "2.0.0"]
        
        for version in versions:
            await db_session.execute(
                """
                INSERT INTO workflow_versions 
                (workflow_id, version, dsl_raw, status)
                VALUES ($1, $2, $3, $4)
                """,
                workflow_id,
                version,
                json.dumps({"version": version}),
                "compiled"
            )
        
        await db_session.commit()
        
        # 버전 목록 조회
        result = await db_session.execute(
            """
            SELECT version, created_at 
            FROM workflow_versions 
            WHERE workflow_id = $1 
            ORDER BY created_at DESC
            """,
            workflow_id
        )
        
        fetched_versions = result.fetchall()
        assert len(fetched_versions) == 4
        assert fetched_versions[0]["version"] == "2.0.0"  # 최신 버전
    
    @pytest.mark.asyncio
    async def test_execution_logging(self, db_session):
        """실행 로깅 테스트"""
        workflow_id = await self._create_test_workflow(db_session)
        
        # 실행 기록
        execution_data = {
            "workflow_id": workflow_id,
            "version": "1.0.0",
            "tenant_id": "tenant_123",
            "session_id": "session_456",
            "status": "running",
            "request_payload": json.dumps({"message": "test"})
        }
        
        result = await db_session.execute(
            """
            INSERT INTO executions 
            (workflow_id, version, tenant_id, session_id, status, request_payload)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING execution_id, started_at
            """,
            execution_data["workflow_id"],
            execution_data["version"],
            execution_data["tenant_id"],
            execution_data["session_id"],
            execution_data["status"],
            execution_data["request_payload"]
        )
        
        execution = result.fetchone()
        execution_id = execution["execution_id"]
        
        # 노드 실행 기록
        nodes = ["intent_1", "llm_1", "api_1"]
        for node_id in nodes:
            await db_session.execute(
                """
                INSERT INTO node_executions
                (execution_id, node_id, node_type, status, duration_ms)
                VALUES ($1, $2, $3, $4, $5)
                """,
                execution_id,
                node_id,
                node_id.split("_")[0],
                "completed",
                100 + nodes.index(node_id) * 50
            )
        
        # 실행 완료 업데이트
        await db_session.execute(
            """
            UPDATE executions 
            SET status = 'completed',
                completed_at = NOW(),
                latency_ms = 350,
                tokens_in = 100,
                tokens_out = 150
            WHERE execution_id = $1
            """,
            execution_id
        )
        
        await db_session.commit()
        
        # 실행 통계 조회
        result = await db_session.execute(
            """
            SELECT 
                COUNT(*) as total_nodes,
                AVG(duration_ms) as avg_duration,
                SUM(duration_ms) as total_duration
            FROM node_executions
            WHERE execution_id = $1
            """,
            execution_id
        )
        
        stats = result.fetchone()
        assert stats["total_nodes"] == 3
        assert stats["total_duration"] == 350  # 100 + 150 + 200
    
    @pytest.mark.asyncio
    async def test_environment_routing(self, db_session):
        """환경별 라우팅 테스트"""
        workflow_id = await self._create_test_workflow(db_session)
        
        # 환경별 버전 설정
        environments = [
            ("development", "2.0.0-dev"),
            ("staging", "1.1.0"),
            ("production", "1.0.0")
        ]
        
        for env, version in environments:
            await db_session.execute(
                """
                INSERT INTO env_routes (workflow_id, environment, active_version)
                VALUES ($1, $2, $3)
                ON CONFLICT (workflow_id, environment) 
                DO UPDATE SET active_version = $3
                """,
                workflow_id,
                env,
                version
            )
        
        await db_session.commit()
        
        # 프로덕션 버전 조회
        result = await db_session.execute(
            """
            SELECT active_version 
            FROM env_routes 
            WHERE workflow_id = $1 AND environment = 'production'
            """,
            workflow_id
        )
        
        prod_version = result.fetchone()
        assert prod_version["active_version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, db_session):
        """감사 로깅 테스트"""
        workflow_id = await self._create_test_workflow(db_session)
        
        # 감사 로그 기록
        audit_log = {
            "entity_type": "workflow",
            "entity_id": workflow_id,
            "action": "update",
            "user_id": "admin_user",
            "user_email": "admin@example.com",
            "user_role": "admin",
            "ip_address": "192.168.1.1",
            "changes": json.dumps({
                "description": {
                    "old": "원래 설명",
                    "new": "새로운 설명"
                }
            })
        }
        
        await db_session.execute(
            """
            INSERT INTO audit_logs 
            (entity_type, entity_id, action, user_id, user_email, 
             user_role, ip_address, changes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            audit_log["entity_type"],
            audit_log["entity_id"],
            audit_log["action"],
            audit_log["user_id"],
            audit_log["user_email"],
            audit_log["user_role"],
            audit_log["ip_address"],
            audit_log["changes"]
        )
        
        await db_session.commit()
        
        # 감사 로그 조회
        result = await db_session.execute(
            """
            SELECT * FROM audit_logs 
            WHERE entity_id = $1 
            ORDER BY created_at DESC
            """,
            workflow_id
        )
        
        logs = result.fetchall()
        assert len(logs) > 0
        assert logs[0]["action"] == "update"
        assert logs[0]["user_role"] == "admin"
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, db_session):
        """성능 메트릭 조회 테스트"""
        workflow_id = await self._create_test_workflow(db_session)
        
        # 여러 실행 기록 생성
        for i in range(10):
            await db_session.execute(
                """
                INSERT INTO executions 
                (workflow_id, version, status, latency_ms, tokens_in, tokens_out)
                VALUES ($1, '1.0.0', $2, $3, $4, $5)
                """,
                workflow_id,
                "completed" if i < 8 else "failed",
                1000 + i * 100,
                50 + i * 10,
                100 + i * 20
            )
        
        await db_session.commit()
        
        # 성능 통계 조회
        result = await db_session.execute(
            """
            SELECT 
                COUNT(*) as total_executions,
                COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
                AVG(latency_ms) as avg_latency,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
                SUM(tokens_in + tokens_out) as total_tokens
            FROM executions
            WHERE workflow_id = $1
            """,
            workflow_id
        )
        
        metrics = result.fetchone()
        assert metrics["total_executions"] == 10
        assert metrics["successful_executions"] == 8
        assert metrics["avg_latency"] == 1450  # 평균 레이턴시
        assert metrics["total_tokens"] > 0
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_session):
        """트랜잭션 롤백 테스트"""
        workflow_id = await self._create_test_workflow(db_session)
        
        try:
            async with db_session.begin():
                # 버전 추가
                await db_session.execute(
                    """
                    INSERT INTO workflow_versions 
                    (workflow_id, version, dsl_raw, status)
                    VALUES ($1, $2, $3, $4)
                    """,
                    workflow_id,
                    "3.0.0",
                    json.dumps({"test": "data"}),
                    "uploaded"
                )
                
                # 의도적 오류 발생
                raise Exception("Rollback test")
        
        except Exception:
            pass  # 예외 무시
        
        # 롤백 확인
        result = await db_session.execute(
            """
            SELECT COUNT(*) as count 
            FROM workflow_versions 
            WHERE workflow_id = $1 AND version = '3.0.0'
            """,
            workflow_id
        )
        
        count = result.fetchone()
        assert count["count"] == 0  # 롤백되어 없어야 함
    
    async def _create_test_workflow(self, db_session):
        """테스트용 워크플로우 생성 헬퍼"""
        result = await db_session.execute(
            """
            INSERT INTO workflows (namespace, name, created_by)
            VALUES ('test', 'test_workflow', 'test_user')
            RETURNING id
            """
        )
        workflow = result.fetchone()
        await db_session.commit()
        return workflow["id"]


class TestRedisIntegration:
    """Redis 통합 테스트"""
    
    @pytest.fixture
    async def redis_client(self):
        """Redis 클라이언트"""
        import aioredis
        
        redis = await aioredis.create_redis_pool(
            settings.REDIS_URL,
            minsize=1,
            maxsize=10
        )
        
        yield redis
        
        redis.close()
        await redis.wait_closed()
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, redis_client):
        """캐시 작업 테스트"""
        # SET
        await redis_client.set("test_key", "test_value", expire=60)
        
        # GET
        value = await redis_client.get("test_key")
        assert value == b"test_value"
        
        # EXISTS
        exists = await redis_client.exists("test_key")
        assert exists == 1
        
        # DELETE
        await redis_client.delete("test_key")
        exists = await redis_client.exists("test_key")
        assert exists == 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, redis_client):
        """세션 관리 테스트"""
        session_id = "session_123"
        session_data = {
            "user_id": "user_456",
            "workflow_id": "workflow_789",
            "context": {"step": 1}
        }
        
        # 세션 저장
        await redis_client.setex(
            f"session:{session_id}",
            3600,  # 1시간 TTL
            json.dumps(session_data)
        )
        
        # 세션 조회
        stored_data = await redis_client.get(f"session:{session_id}")
        retrieved = json.loads(stored_data)
        assert retrieved["user_id"] == "user_456"
        
        # TTL 확인
        ttl = await redis_client.ttl(f"session:{session_id}")
        assert ttl > 0 and ttl <= 3600
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, redis_client):
        """Rate Limiting 테스트"""
        user_id = "test_user"
        key = f"rate_limit:{user_id}"
        
        # 초기 카운터
        await redis_client.setex(key, 60, 0)
        
        # 요청 카운트 증가
        for i in range(5):
            count = await redis_client.incr(key)
            assert count == i + 1
        
        # 한계 확인
        current = await redis_client.get(key)
        assert int(current) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])