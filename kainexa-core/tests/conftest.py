# tests/conftest.py
import os
import pytest
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient

from src.api.main import app

@pytest.fixture(scope="session")
def anyio_backend():
    # pytest-asyncio/anyio가 asyncio 모드로 동작하도록 고정
    return "asyncio"

@pytest.fixture
async def async_client():
    """
    올바른 형태의 'async fixture':
    - 반드시 'async def' + 'yield client'
    - httpx ASGITransport로 lifespan='on' -> startup/shutdown 실행
    """
    os.environ["TESTING"] = "true"
    transport = ASGITransport(app=app, lifespan="on")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client  # <-- 'return' 금지! 반드시 'yield'

@pytest.fixture
def test_client():
    os.environ["TESTING"] = "true"
    with TestClient(app) as client:
        yield client

@pytest.fixture
def auth_headers():
    # 통합 테스트에서 기대하는 헤더 셋
    return {
        "Authorization": "Bearer test-jwt-token",
        "X-API-Key": "test-api-key",
    }
