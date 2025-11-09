from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class SearchHit(BaseModel):
    id: str
    law_name: Optional[str] = None
    clause_id: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None

    # 정렬에 쓰이는 최종 score (RRF 또는 리랭크 점수)
    score: float

    # ⬇️ 추가: 원점수들 (있으면 내려주고, 없으면 null)
    bm25_score: Optional[float] = None
    dense_cosine: Optional[float] = None


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
