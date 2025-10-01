# src/api/routes/metrics.py
from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])

@router.get("/workflows")
async def metrics_workflows(
    start_date: str = Query(...),
    end_date: str = Query(...),
    granularity: str = Query("daily"),
):
    return {
        "period": {"start_date": start_date, "end_date": end_date, "granularity": granularity},
        "metrics": {
            "total_executions": 10,
            "success_rate": 0.9,
            "avg_latency": 120,
            "total_tokens": 1234,
        },
    }
