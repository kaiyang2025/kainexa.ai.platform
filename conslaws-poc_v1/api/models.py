from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SearchHit(BaseModel):
    id: str
    law_name: str
    clause_id: str
    title: str
    text: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]

class AnswerRequest(BaseModel):
    query: str
    k: int = 8
    rerank: bool = True
    include_context: bool = True
    gen_backend: Optional[str] = None
    gen_model: Optional[str] = None
    # 리랭크 후보 폭 (k의 배수). None이면 서버에서 1.0으로 처리
    cand_factor: Optional[float] = None

class Citation(BaseModel):
    law: str
    clause_id: str

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    contexts: Optional[List[Dict[str, Any]]] = None
