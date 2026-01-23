# -*- coding: utf-8 -*-
"""
search_utils.py
- 검색 엔진: Kiwi+BM25(키워드), BGE-M3(벡터), CrossEncoder(리랭킹)
- 답변 생성: 내부 구축형 LLM (OpenAI 호환 API)
"""

import os
import json
import re
import pathlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv

# ---------------------------------------------------------
# [라이브러리] 설치 필요:
# pip install rank_bm25 kiwipiepy chromadb sentence_transformers langchain_openai langchain_core
# ---------------------------------------------------------
import torch
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. 설정 및 경로 (수정됨)
# ---------------------------------------------------------
# search_utils.py는 'api' 폴더 안에 있으므로,
# 부모의 부모 디렉토리가 프로젝트 루트가 됩니다.
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # /home/kai/.../conslaws-poc_v1/

# .env 로드
load_dotenv(PROJECT_ROOT / ".env", override=True)

# 경로 설정
DATA_PROC_DIR = PROJECT_ROOT / "data_proc"
INDEX_DIR = PROJECT_ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"

DATA_FILE = DATA_PROC_DIR / "law_clauses.jsonl"
GLOSSARY_FILE = PROJECT_ROOT / "data_raw" / "glossary.csv"
CHROMA_COLLECTION_NAME = "laws_bge_m3_v2"

# LLM 연결 정보 (환경변수 필수)
LLM_BASE_URL = os.getenv("BASE_URL") 
LLM_API_KEY = os.getenv("API_KEY")

# ---------------------------------------------------------
# 2. 유틸리티 함수: RRF 결합
# ---------------------------------------------------------
def rrf_merge(bm25_list: List[dict], dense_list: List[dict], k=60) -> List[dict]:
    """Reciprocal Rank Fusion"""
    ranks = defaultdict(lambda: {"bm25_rank": None, "dense_rank": None, "fused_score": 0.0, "doc_data": {}})
    
    for i, item in enumerate(bm25_list):
        # ID가 없으면 생성
        rid = str(item.get("id", f"{item.get('law_name')}|{item.get('clause_id')}"))
        ranks[rid]["bm25_rank"] = i + 1
        ranks[rid]["doc_data"] = item

    for i, item in enumerate(dense_list):
        rid = str(item.get("id", f"{item.get('law_name')}|{item.get('clause_id')}"))
        ranks[rid]["dense_rank"] = i + 1
        if not ranks[rid]["doc_data"]: 
            ranks[rid]["doc_data"] = item

    fused_results = []
    for rid, info in ranks.items():
        score = 0.0
        if info["bm25_rank"]: score += 1.0 / (k + info["bm25_rank"])
        if info["dense_rank"]: score += 1.0 / (k + info["dense_rank"])
        
        doc = info["doc_data"].copy()
        doc["fused_score"] = score
        fused_results.append(doc)

    fused_results.sort(key=lambda x: -x["fused_score"])
    return fused_results

# ---------------------------------------------------------
# 3. 검색 엔진 클래스
# ---------------------------------------------------------
class ConstructionBM25:
    """[Keyword Search] Kiwi + BM25"""
    CIRCLED_TO_ARABIC = dict(zip("①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳", map(str, range(1, 21))))

    def __init__(self, glossary_path: pathlib.Path = None):
        self.kiwi = Kiwi()
        self.bm25 = None
        self.docs = []
        if glossary_path and glossary_path.exists():
            self._load_glossary(glossary_path)

    def _load_glossary(self, path: pathlib.Path):
        try:
            df = pd.read_csv(path)
            terms = df['term'].dropna().astype(str).str.strip().unique().tolist()
            for term in terms:
                if term: self.kiwi.add_user_word(term, tag='NNG', score=10)
        except Exception:
            pass

    def _normalize(self, text: str) -> str:
        text = re.sub(r"제\s*(\d+)\s*조\s*의\s*(\d+)", r"제\1조의\2", text)
        text = re.sub(r"제\s*(\d+)\s*조", r"제\1조", text)
        text = re.sub(r"제\s*(\d+)\s*항", r"제\1항", text)
        for char, digit in self.CIRCLED_TO_ARABIC.items():
            text = text.replace(char, digit)
        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        results = self.kiwi.analyze(self._normalize(text))
        for res in results:
            for token, tag, _, _ in res[0]:
                if tag.startswith('N') or tag in ('SL', 'SN', 'XPN', 'XR'):
                    tokens.append(token)
        return tokens

    def fit(self, jsonl_path: pathlib.Path):
        # 경로 객체를 문자열로 변환하여 출력 (디버깅 용이)
        if not jsonl_path.exists():
            print(f"[BM25] ❌ 데이터 파일 없음: {jsonl_path.absolute()}")
            return
        
        print(f"[BM25] 데이터 로드 중: {jsonl_path.name}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f if line.strip()]
        
        corpus_tokens = [self.tokenize(f'{d.get("clause_id", "")} {d.get("text", "")}') for d in self.docs]
        self.bm25 = BM25Okapi(corpus_tokens, k1=1.2, b=0.75)
        print(f"[BM25] 인덱싱 완료 ({len(self.docs)}건)")

    def search(self, query: str, topn: int = 10) -> List[Dict]:
        if not self.bm25: return []
        scores = self.bm25.get_scores(self.tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:topn]
        return [dict(self.docs[i], bm25_score=float(scores[i])) for i in top_indices if scores[i] > 0]

class DenseRetriever:
    """[Vector Search] ChromaDB + BGE-M3"""
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Dense] 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.collection = None

    def load_chroma(self, db_path: pathlib.Path, collection_name: str) -> bool:
        if not db_path.exists(): 
            print(f"[Dense] ❌ ChromaDB 경로 없음: {db_path.absolute()}")
            return False
        try:
            client = chromadb.PersistentClient(path=str(db_path), settings=Settings(allow_reset=True))
            self.collection = client.get_collection(collection_name)
            print(f"[Dense] ChromaDB 로드 성공 ({self.collection.count()}건)")
            return True
        except Exception as e:
            print(f"[Dense] 로드 실패: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.collection: return []
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        results = self.collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k, include=["documents", "metadatas", "distances"])
        
        parsed = []
        if results['ids']:
            for i, doc_id in enumerate(results['ids'][0]):
                parsed.append({
                    "id": doc_id,
                    "text": results['documents'][0][i],
                    "dense_score": float(1.0 - results['distances'][0][i]),
                    **results['metadatas'][0][i]
                })
        return parsed

class CrossEncoderReRanker:
    """[Re-Ranker] BGE-Reranker-v2-m3"""
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Rerank] 모델 로딩: {model_name}")
        self.model = CrossEncoder(model_name, device=self.device, max_length=512, trust_remote_code=True)

    def rerank(self, query: str, candidates: List[dict], top_n=5) -> List[dict]:
        if not candidates: return []
        targets = candidates[:50]
        pairs = [[query, f"{doc.get('law_name','')} {doc.get('text','')[:512]}"] for doc in targets]
        
        try:
            scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
            scores = 1 / (1 + np.exp(-scores)) # Sigmoid
            for i, s in enumerate(scores):
                targets[i]["fused_score"] = float(s)
            targets.sort(key=lambda x: -x["fused_score"])
            return targets[:top_n]
        except Exception:
            return candidates[:top_n]

# ---------------------------------------------------------
# 4. 파이프라인 싱글톤
# ---------------------------------------------------------
class RAGPipeline:
    def __init__(self):
        print(">>> [System] 파이프라인 초기화...")
        # 데이터 파일 위치 출력 (디버깅용)
        print(f"   - PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"   - DATA_FILE: {DATA_FILE}")
        
        self.bm25 = ConstructionBM25(glossary_path=GLOSSARY_FILE)
        self.bm25.fit(DATA_FILE)
        
        self.dense = DenseRetriever()
        self.dense.load_chroma(CHROMA_DIR, CHROMA_COLLECTION_NAME)
        
        self.reranker = CrossEncoderReRanker()
        print(">>> [System] 준비 완료.")

    def search(self, query: str, k: int = 5, bm25_k: int = 20, dense_k: int = 20, rerank_input_k: int = 50, rerank: bool = True) -> List[Dict]:
        
        # 1. 1차 검색 (Retrieval) - 각각 설정된 개수만큼 가져오기
        bm25_res = self.bm25.search(query, topn=bm25_k)
        dense_res = self.dense.search(query, top_k=dense_k)
        
        # 2. RRF 결합
        fused = rrf_merge(bm25_res, dense_res, k=60)
        
        # 3. Re-ranking
        if rerank:
            # Reranker에게 보낼 개수(rerank_input_k)만큼 자르고, 최종 k개 반환
            return self.reranker.rerank(query, fused[:rerank_input_k], top_n=k)
        else:
            return fused[:k]

_PIPELINE = None
def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None: _PIPELINE = RAGPipeline()
    return _PIPELINE

# ---------------------------------------------------------
# 5. API 함수 (외부 호출용)
# ---------------------------------------------------------
def search_docs(query: str, k: int = 5, bm25_k: int = 20, dense_k: int = 20, rerank_input_k: int = 50, rerank: bool = True) -> List[Dict]:
    return get_pipeline().search(query, k, bm25_k, dense_k, rerank_input_k, rerank)


def generate_answer(query: str, contexts: List[Dict], backend: str = "custom", model: str = "openai/gpt-oss-120b") -> str:
    """
    내부 LLM 서버를 사용하여 답변 생성
    """
    if not contexts:
        return "문서에서 정보를 찾을 수 없습니다."

    joined_context = "\n\n".join([f"[{i+1}] {doc.get('law_name')} {doc.get('clause_id')}\n{doc.get('text')}" for i, doc in enumerate(contexts)])
    
    if not LLM_BASE_URL:
        return "오류: BASE_URL 환경변수가 설정되지 않았습니다."
        
    try:
        llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY if LLM_API_KEY else "EMPTY",
            model=model,
            temperature=0.1,
        )

        answer_prompt = ChatPromptTemplate.from_template("""
You are a legal professional specializing in Korean construction law.
[Answer] the user's [Question] based ONLY on the provided [Document].
Your [Answer] must refer only to the [Document] provided.
Your  [Answer] must be written in Korean.

[Document]{context}
[Question]{question}

[Answer]
""")
        answer_chain = answer_prompt | llm | StrOutputParser()
        response = answer_chain.invoke({"question": query, "context": joined_context[:15000]})
        return response

    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# ---------------------------------------------------------
# 테스트 코드
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"BASE_URL: {LLM_BASE_URL}")
    test_q = "하도급대금 직접지급 요건은?"
    
    docs = search_docs(test_q, k=3)
    print(f"검색된 문서: {len(docs)}건")
    
    print("답변 생성 중...")
    ans = generate_answer(test_q, docs)
    print("="*50)
    print(ans)
    print("="*50)