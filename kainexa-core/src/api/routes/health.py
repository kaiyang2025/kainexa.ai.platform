# src/api/routes/health.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db
from src.core.redis_client import redis_client
import httpx

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "kainexa-core",
        "version": "0.1.0"
    }

@router.get("/health/detailed")
async def detailed_health(db: AsyncSession = Depends(get_db)):
    """Detailed health check with dependency status"""
    health_status = {
        "api": "healthy",
        "database": "unknown",
        "redis": "unknown",
        "qdrant": "unknown"
    }
    
    # Check database
    try:
        await db.execute("SELECT 1")
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        await redis_client.redis.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
    
    # Check Qdrant
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/health")
            if response.status_code == 200:
                health_status["qdrant"] = "healthy"
    except Exception as e:
        health_status["qdrant"] = f"unhealthy: {str(e)}"
    
    overall_health = all(v == "healthy" for v in health_status.values())
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "components": health_status
    }