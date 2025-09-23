# src/api/main_simple.py 생성

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import logging
import sys
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Kainexa Core API")
    yield
    # Shutdown
    logger.info("Shutting down Kainexa Core API")

# Create FastAPI app
app = FastAPI(
    title="Kainexa Core API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# Routes
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Kainexa AI Platform API",
        "version": "0.1.0",
        "docs": "/api/v1/docs"
    }

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="kainexa-core",
        version="0.1.0"
    )

@app.get("/api/v1/health/detailed", tags=["Health"])
async def detailed_health():
    """Detailed health check with service status"""
    health_status = {
        "api": "healthy",
        "timestamp": "2025-09-23T10:00:00Z"
    }
    
    # GPU 상태 확인
    try:
        import torch
        health_status["cuda_available"] = torch.cuda.is_available()
        health_status["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            health_status["gpu_0"] = torch.cuda.get_device_name(0)
    except ImportError:
        health_status["cuda_available"] = False
        health_status["gpu_count"] = 0
    
    # Database 연결 테스트 (비동기)
    try:
        import asyncpg
        import asyncio
        
        async def check_db():
            try:
                conn = await asyncpg.connect(
                    'postgresql://kainexa:password@localhost:5432/kainexa_db',
                    timeout=2
                )
                await conn.close()
                return "healthy"
            except Exception as e:
                return f"error: {str(e)[:50]}"
        
        health_status["database"] = await check_db()
    except Exception as e:
        health_status["database"] = "not configured"
    
    # Redis 연결 테스트
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        health_status["redis"] = "healthy"
    except Exception:
        health_status["redis"] = "not available"
    
    # Qdrant 연결 테스트
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:6333/")
            health_status["qdrant"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        health_status["qdrant"] = "not available"
    
    return health_status

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Simple chat endpoint for testing"""
    # 간단한 에코 응답
    response_text = f"Echo: {request.message}"
    
    # 특정 키워드에 대한 응답
    if "안녕" in request.message:
        response_text = "안녕하세요! Kainexa AI입니다. 무엇을 도와드릴까요?"
    elif "도움" in request.message or "help" in request.message.lower():
        response_text = "저는 Kainexa AI 어시스턴트입니다. 다양한 질문에 답변할 수 있습니다."
    
    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        status="success"
    )

@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """List available AI models"""
    return {
        "models": [
            {"id": "solar-10.7b", "name": "Solar 10.7B", "status": "available"},
            {"id": "gpt2", "name": "GPT-2", "status": "available"},
            {"id": "polyglot", "name": "Polyglot-Ko", "status": "planned"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
