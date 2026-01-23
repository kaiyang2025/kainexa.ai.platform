# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

# 변경된 search_utils에서 필요한 함수들만 임포트
from .search_utils import search_docs, generate_answer

app = FastAPI(title="Construction Law RAG API")

# 요청 데이터 모델 (Streamlit에서 보내는 데이터와 일치)
class QueryRequest(BaseModel):
    query: str
    k: int = 5                  # Final Output 개수
    bm25_k: int = 20            # [추가] BM25 후보 수
    dense_k: int = 20           # [추가] Dense 후보 수
    rerank_input_k: int = 50    # [추가] Rerank 입력 후보 수
    rerank: bool = True
    include_context: bool = True
    gen_backend: str = "custom"
    gen_model: str = "openai/gpt-oss-120b"
    # cand_factor 제거

# 응답 데이터 모델
class AnswerResponse(BaseModel):
    answer: str
    contexts: List[dict] = []
    latency: float = 0.0

@app.post("/answer", response_model=AnswerResponse)
async def get_answer(req: QueryRequest):
    start_time = time.time()
    try:
        # [수정] search_docs 호출 시 파라미터 전달 변경
        contexts = search_docs(
            query=req.query,
            k=req.k,
            bm25_k=req.bm25_k,
            dense_k=req.dense_k,
            rerank_input_k=req.rerank_input_k,
            rerank=req.rerank
        )

        # 2. 답변 생성 (search_utils.py의 함수 호출)
        # 내부 LLM 서버(gpt-oss-120b)를 사용하여 답변 생성
        answer_text = generate_answer(
            query=req.query,
            contexts=contexts,
            backend=req.gen_backend,
            model=req.gen_model
        )
        
        elapsed = time.time() - start_time
        
        return AnswerResponse(
            answer=answer_text,
            contexts=contexts if req.include_context else [],
            latency=elapsed
        )

    except Exception as e:
        print(f"[API Error] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 로컬 테스트용 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)