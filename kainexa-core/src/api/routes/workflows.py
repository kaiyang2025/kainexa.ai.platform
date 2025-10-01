# src/api/routes/workflows.py
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/v1", tags=["workflows"])

def require_api_key(x_api_key: str | None) -> None:
    if x_api_key not in {"valid-api-key", "test-api-key"}:
        raise HTTPException(status_code=401, detail="invalid api key")

class UploadBody(BaseModel):
    dsl_content: str

@router.post("/workflows/upload")
async def upload_workflow(body: UploadBody, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    return {"workflow_id": "test-workflow-id", "status": "uploaded", "version": "1.0.0"}

@router.post("/workflows/compile")
async def compile_workflow(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    return {"status": "compiled"}

class ExecuteBody(BaseModel):
    input: dict
    context: dict | None = None

@router.post("/workflow/{namespace}/{name}/execute")
async def execute_workflow(namespace: str, name: str, body: ExecuteBody, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    exec_id = f"exec_{uuid.uuid4().hex[:8]}"
    return {
        "status": "completed",
        "execution_id": exec_id,
        "outputs": {"content": f"echo: {body.input.get('message', '')}"},
        "metadata": {"model": "stub", "latency_ms": 10},
    }

@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    return {"execution_id": execution_id, "status": "completed"}

@router.get("/workflows")
async def list_workflows(x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    return [{"id": "test-workflow", "namespace": "test", "name": "test_workflow", "version": "1.0.0"}]

@router.get("/workflows/{workflow_id}/versions")
async def get_versions(workflow_id: str):
    return {
        "workflow_id": workflow_id,
        "versions": [
            {"version": "1.2.0", "created_at": datetime(2024,1,15).isoformat()},
            {"version": "1.1.0", "created_at": datetime(2024,1,10).isoformat()},
            {"version": "1.0.0", "created_at": datetime(2024,1,5).isoformat()},
        ],
    }
