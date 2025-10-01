from __future__ import annotations
from fastapi import APIRouter, Query
from pydantic import BaseModel
import uuid

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

class UploadDocumentRequest(BaseModel):
    content: str
    metadata: dict | None = None

class UploadDocumentResponse(BaseModel):
    doc_id: str
    status: str = "stored"

@router.post("/documents", response_model=UploadDocumentResponse)
async def upload_document(req: UploadDocumentRequest):
    return UploadDocumentResponse(doc_id=str(uuid.uuid4()))

@router.get("/search")
async def search(q: str = Query(..., min_length=1), top_k: int = 5):
    # 간단 스텁 결과
    return {
        "query": q,
        "results": [
            {"id": f"doc_{i}", "score": 1.0 - i * 0.1, "content": f"Result {i} for {q}"}
            for i in range(max(0, top_k))
        ],
    }
