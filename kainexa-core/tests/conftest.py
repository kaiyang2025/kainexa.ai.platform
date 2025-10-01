# tests/conftest.py
import os
import pytest
from httpx import AsyncClient
from asgi_lifespan import LifespanManager
from starlette.testclient import TestClient

from src.api.main import app

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture
async def async_client():
    os.environ["TESTING"] = "true"
    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

@pytest.fixture
def test_client():
    os.environ["TESTING"] = "true"
    with TestClient(app) as client:
        yield client

# (있다면 테스트 파일의 auth_headers와 충돌하지 않으니 이건 생략 가능)
# @pytest.fixture
# def auth_headers():
#     return {"Authorization": "Bearer test-jwt-token", "X-API-Key": "test-api-key"}
