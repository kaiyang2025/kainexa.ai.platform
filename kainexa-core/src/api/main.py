# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
from prometheus_fastapi_instrumentator import Instrumentator

from src.core.config import settings
from src.core.database import engine
from src.core.redis_client import redis_client
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import (
     health, conversations, knowledge, analytics,
     workflows, auth, users, metrics, ws,
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Kainexa Core API", version=settings.APP_VERSION)

    # 테스트 환경에서는 연결/종료를 생략하거나 실패 무시
    try:
        if not getattr(settings, "TESTING", False):
            await redis_client.connect()
    except Exception as e:
        logger.warning("Redis connect skipped in testing or failed: %s", e)

    yield

    try:
        if not getattr(settings, "TESTING", False):
            await redis_client.disconnect()
            await engine.dispose()
    except Exception as e:
        logger.warning("Shutdown cleanup skipped in testing or failed: %s", e)

    logger.info("Shutting down Kainexa Core API")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.add_middleware(
    RateLimitMiddleware,
    max_requests=3 if getattr(settings, "TESTING", False) else 60,
    window_seconds=1 if getattr(settings, "TESTING", False) else 60,
)

# Add Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Include routers
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["health"])
app.include_router(conversations.router, prefix=settings.API_PREFIX, tags=["conversations"])
app.include_router(knowledge.router, prefix=settings.API_PREFIX, tags=["knowledge"])
app.include_router(analytics.router, prefix=settings.API_PREFIX, tags=["analytics"])
app.include_router(workflows.router, prefix=settings.API_PREFIX, tags=["workflows"])
app.include_router(auth.router, prefix=settings.API_PREFIX, tags=["auth"])
app.include_router(users.router, prefix=settings.API_PREFIX, tags=["users"])
app.include_router(metrics.router, prefix=settings.API_PREFIX, tags=["metrics"])
# WebSocket은 prefix 없이도 등록 (테스트는 /ws/... 사용)
app.include_router(ws.router, tags=["ws"])                     # /ws/...
app.include_router(ws.router, prefix=settings.API_PREFIX, tags=["ws"])  # /api/v1/ws/...

@app.get("/health")
async def health_alias():
    return {"status": "ok"}
