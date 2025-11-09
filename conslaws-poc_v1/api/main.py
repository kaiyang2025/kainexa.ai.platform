from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from .models import SearchResponse, SearchHit, AnswerRequest, AnswerResponse, Citation
from .search_utils import Retriever
from .config import SYSTEM_PROMPT, GEN_BACKEND, GEN_MODEL
import os

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = FastAPI(title="Construction-Law-RAG-POC API", version="0.2.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

retriever = Retriever()

def build_prompt(query: str, contexts: List[dict]):
    header = SYSTEM_PROMPT.strip()
    ctx_lines = []
    for c in contexts:
        badge = f"[{c['law_name']} {c['clause_id']}]"
        snippet = c["text"].strip().replace("\n", " ")
        ctx_lines.append(f"- {badge} {snippet}")
    ctx = "\n".join(ctx_lines)
    return f"""{header}

사용자 질문: {query}

컨텍스트:
{ctx}

위 규칙을 따라 한국어로 간결하게 답변하시오.
"""

def call_llm(prompt: str, backend: str, model: str) -> str:
    backend = backend or GEN_BACKEND
    model = model or GEN_MODEL
    if backend == "openai" and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"You are a helpful assistant."},
                      {"role":"user","content":prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content
    return "※ 데모용(dummy): 검색 문맥 기반으로 요약 답변을 제공합니다.\n" + \
           "\n".join([line for line in prompt.splitlines()[:10]])

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search", response_model=SearchResponse)
def search(q: str, k: int = 8, rerank: bool = True, cand_factor: float = 1.0):
    """
    cand_factor: 리랭커에 태울 후보 폭 (k의 배수). 예) 2.0 => Top-2k 리랭크
    """
    hits = retriever.search(q, rerank=rerank, k=k, cand_factor=cand_factor)        
    results = [SearchHit(**h) for h in hits]
    return SearchResponse(query=q, results=results)

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    # req.cand_factor가 None이면 1.0으로 처리
    cf = req.cand_factor if getattr(req, "cand_factor", None) is not None else 1.0
    hits = retriever.search(req.query, rerank=req.rerank, k=req.k, cand_factor=cf)
    prompt = build_prompt(req.query, hits)
    text = call_llm(prompt, req.gen_backend, req.gen_model)
    cits = [Citation(law=h["law_name"], clause_id=h["clause_id"]) for h in hits]
    contexts = hits if req.include_context else None
    return AnswerResponse(answer=text, citations=cits, contexts=contexts)
