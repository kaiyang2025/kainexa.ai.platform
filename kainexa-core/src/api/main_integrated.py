# kainexa-core/src/api/main_integrated.py 수정
"""
통합 API 서버
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.routes.integrated import router as integrated_router
from src.core.config import settings
from src.models.solar_llm import SolarLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Kainexa Integrated API")
    
    # ✅ LLM 싱글턴 예열 & 캐싱
    try:
        if not hasattr(app.state, "llm") or app.state.llm is None:
            llm = SolarLLM()
            llm.load()                 # 모델 1회 로드
            app.state.llm = llm        # 상태에 캐시
            logger.info("SolarLLM preloaded and cached in app.state.llm")
    except Exception as e:
        logger.exception("LLM warm-up failed (routes may lazy-init): %s", e)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Kainexa Integrated API")
    # (선택) 메모리 정리
    try:
        app.state.llm = None
    except Exception:
        pass

app = FastAPI(
    title="Kainexa AI Platform",
    version="1.0.0",
    description="Manufacturing AI Agent Platform",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함
app.include_router(integrated_router)

@app.get("/")
async def root():
    return {
        "name": "Kainexa AI Platform",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "chat": "/api/v1/chat",
            "documents": "/api/v1/documents",
            "scenarios": "/api/v1/scenarios",
            "health": "/api/v1/health/full"
        }
    }

@app.post("/api/v1/workflow/execute")
async def execute_workflow(request: dict):
    """워크플로우 실행"""
    nodes = request.get("nodes", [])
    edges = request.get("edges", [])
    
    print(f"Executing workflow with {len(nodes)} nodes")
    
    return {
        "execution_id": f"exec_{datetime.now().timestamp()}",
        "status": "completed",
        "message": f"워크플로우 실행 완료: {len(nodes)}개 노드 처리",
        "timestamp": datetime.now().isoformat()
    }

# ✅ /health 엔드포인트 추가
@app.get("/health")
async def health_check():
    """간단한 헬스 체크"""
    return {
        "status": "healthy",
        "service": "kainexa-core",
        "llm_loaded": hasattr(app.state, "llm") and app.state.llm is not None
    }

# ✅ API 상태 확인
@app.get("/api/status")
async def api_status():
    """API 상태 확인"""
    llm_status = "loaded" if (hasattr(app.state, "llm") and app.state.llm is not None) else "not_loaded"
    return {
        "status": "operational",
        "version": "1.0.0",
        "llm": llm_status,
        "endpoints": [
            "/api/v1/chat",
            "/api/v1/documents", 
            "/api/v1/scenarios"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)