# main_complete.py - 전체 기능 포함

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Kainexa AI Platform",
    description="AI Agent Platform for Manufacturing and Enterprise",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    intent: Optional[str] = None
    confidence: Optional[float] = None

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

# 루트 엔드포인트
@app.get("/")
async def root():
    return {
        "name": "Kainexa AI Platform",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
            "api_v1": {
                "health": "/api/v1/health",
                "chat": "/api/v1/chat",
                "models": "/api/v1/models"
            }
        }
    }

# Health Check
@app.get("/health", response_model=HealthCheck)
async def health_check():
    services_status = {}
    
    # PostgreSQL 체크
    try:
        import asyncpg
        conn = await asyncpg.connect(
            'postgresql://kainexa:password@localhost:5432/kainexa_db',
            timeout=2
        )
        await conn.close()
        services_status["postgresql"] = "healthy"
    except:
        services_status["postgresql"] = "unavailable"
    
    # Redis 체크
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
        r.ping()
        services_status["redis"] = "healthy"
    except:
        services_status["redis"] = "unavailable"
    
    # Qdrant 체크
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:6333/")
            services_status["qdrant"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services_status["qdrant"] = "unavailable"
    
    # GPU 체크
    try:
        import torch
        if torch.cuda.is_available():
            services_status["gpu"] = f"available ({torch.cuda.device_count()} devices)"
        else:
            services_status["gpu"] = "not available"
    except:
        services_status["gpu"] = "not configured"
    
    return HealthCheck(
        status="healthy" if "healthy" in services_status.values() else "degraded",
        timestamp=datetime.now().isoformat(),
        services=services_status
    )

# Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(f"Chat request: {request.message[:50]}...")
    
    # 간단한 응답 생성 (실제로는 AI 모델 사용)
    response_text = process_message(request.message)
    
    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        timestamp=datetime.now().isoformat(),
        intent="general",
        confidence=0.95
    )

# API v1 endpoints
@app.get("/api/v1/health")
async def health_v1():
    return await health_check()

@app.post("/api/v1/chat")
async def chat_v1(request: ChatRequest):
    return await chat(request)

@app.get("/api/v1/models")
async def list_models():
    return {
        "models": [
            {
                "id": "solar-10.7b",
                "name": "Solar 10.7B (Korean)",
                "status": "available",
                "description": "Korean optimized LLM"
            },
            {
                "id": "gpt-2",
                "name": "GPT-2",
                "status": "available",
                "description": "General purpose model"
            }
        ]
    }

@app.get("/api/v1/sessions")
async def list_sessions():
    return {
        "sessions": [
            {"id": "default", "created": "2025-09-23T10:00:00", "messages": 0}
        ]
    }

# Helper function
def process_message(message: str) -> str:
    """간단한 메시지 처리"""
    message_lower = message.lower()
    
    if "안녕" in message:
        return "안녕하세요! Kainexa AI입니다. 무엇을 도와드릴까요?"
    elif "매출" in message or "판매" in message:
        return "매출 관련 문의시군요. 어떤 기간의 매출 정보가 필요하신가요?"
    elif "생산" in message or "제조" in message:
        return "생산 현황에 대해 알려드리겠습니다. 구체적으로 어떤 정보가 필요하신가요?"
    elif "help" in message_lower or "도움" in message:
        return "저는 제조업 특화 AI 어시스턴트입니다. 생산, 품질, 매출 등 다양한 업무를 도와드릴 수 있습니다."
    else:
        return f"'{message}'에 대한 답변을 준비중입니다. 좀 더 구체적으로 말씀해 주시겠어요?"

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
