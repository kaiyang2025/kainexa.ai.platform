from __future__ import annotations
from fastapi import APIRouter
import os
from datetime import datetime

router = APIRouter(prefix="/api/v1/health", tags=["health"])

@router.get("", name="health:ready")
async def health_ready():
    return {"status": "ok"}

@router.get("/full", name="health:full")
async def health_full():
    # 필요한 값은 환경변수/상수로 대체
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": os.getenv("APP_VERSION", "0.0.0"),
        "services": {
            "db": "ok",
            "vector": "ok",
            "cache": "ok",
        },
    }
