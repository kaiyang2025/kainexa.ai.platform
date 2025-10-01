from __future__ import annotations
from fastapi import APIRouter
from datetime import datetime

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

@router.get("/metrics")
async def get_metrics():
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "requests": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms_avg": 0,
    }
