"""
통합 API 서버
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.routes.integrated import router as integrated_router
from src.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Kainexa Integrated API")
    yield
    # Shutdown
    logger.info("Shutting down Kainexa Integrated API")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)