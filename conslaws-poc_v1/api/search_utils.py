# -*- coding: utf-8 -*-
"""
search_utils.py
- 하이브리드 검색(BM25 + Dense/FAISS) + RRF/가중합 결합 + (선택) CrossEncoder 리랭크
- 인덱스 경로: ./index/{faiss.index, meta.json, docs.jsonl}
- OpenSearch 문서 스키마: {id, law_name, clause_id, title, text, ...}
"""
from __future__ import annotations


import json
import pathlib
from typing import List, Dict, Any, Optional

import faiss  # type: ignore
import numpy as np
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer, CrossEncoder


from .config import (
    OPENSEARCH_URL, OPENSEARCH_INDEX,
    EMBED_MODEL, RERANK_MODEL,
    BM25_K, DENSE_K, FINAL_K, RERANK_CAND_FACTOR,
    LAMBDA_BM25,
    LAMBDA_DENSE,
)

# 경로 상수
ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"
DOCS_PATH = INDEX_DIR / "docs.jsonl"
META_PATH = INDEX_DIR / "meta.json"
FAISS_PATH = INDEX_DIR / "faiss.index"

class Retriever:    
    """
    하이브리드 검색기
    - OpenSearch(BM25) + FAISS(Dense)
    - 결합: RRF(기본) 또는 가중합(alpha)
    - 리랭크: CrossEncoder(옵션)
    """
    def __init__(self,
        opensearch_url: Optional[str] = None,
        index_name: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        faiss_dir: Optional[str] = None,
        reranker_name: Optional[str] = None,
    ):
        # ---- 설정 ----
        self.opensearch_url = opensearch_url or OPENSEARCH_URL
        self.index_name = index_name or OPENSEARCH_INDEX
        self.embed_model_name = embed_model_name or EMBED_MODEL
        
        # ---- OpenSearch ----
        #self.client = OpenSearch(OPENSEARCH_URL, timeout=60)
        self.client = OpenSearch(self.opensearch_url, timeout=60)
        
        
         # ---- 임베딩/FAISS ----        
        #elf.embedder = SentenceTransformer(EMBED_MODEL)
        self.embedder = SentenceTransformer(self.embed_model_name)        
        #self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        base = pathlib.Path(faiss_dir) if faiss_dir else INDEX_DIR
        faiss_path = base / "faiss.index"
        meta_path = base / "meta.json"
        docs_path = base / "docs.jsonl"
        
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        if not docs_path.exists():
            raise FileNotFoundError(f"Docs file not found: {docs_path}")
        
        self.index = faiss.read_index(str(faiss_path))
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))        
        #self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        
        # 문서 맵(id -> record)
        self.docs: Dict[str, Dict[str, Any]] = {}
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                self.docs[r["id"]] = r
        
        #self.docs = {}
        #with open(DOCS_PATH, "r", encoding="utf-8") as f:
        #    for line in f:
        #        r = json.loads(line)
        #        self.docs[r["id"]] = r
        #self._reranker = None
        
         # ---- 리랭커 (지연 로드) ----
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_name = reranker_name or RERANK_MODEL
    
    # ---------- 내부 유틸 ----------
    
    @property
    def reranker(self) -> Optional[CrossEncoder]:
        """CrossEncoder를 필요할 때만 로드"""
        if self._reranker is None and self._reranker_name:
            try:
                self._reranker = CrossEncoder(self._reranker_name)
            except Exception:
                # 리랭커 로딩 실패 시 비활성
                self._reranker = None
        return self._reranker

    def _bm25(self, q: str, k: int) -> List[Dict[str, Any]]:
        """OpenSearch BM25 상위 k 반환"""
        body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": q,
                    "fields": ["text^2", "title"],
                    "type": "best_fields",
                }
            },
            "_source": True,
        }
        resp = self.client.search(index=self.index_name, body=body)
        hits: List[Dict[str, Any]] = []
        for h in resp.get("hits", {}).get("hits", []):
            src = h.get("_source", {})
            # OpenSearch에서 가져온 필드 우선, 없으면 docs.jsonl 보조
            doc_id = h.get("_id")
            rec = {
                "id": doc_id,
                "score": float(h.get("_score", 0.0)),
                **({} if src is None else src),
            }
            # 누락 필드 보강
            if doc_id in self.docs:
                for key in ("law_name", "clause_id", "title", "text"):
                    rec.setdefault(key, self.docs[doc_id].get(key))
            hits.append(rec)
        return hits

    def _dense(self, q: str, k: int) -> List[Dict[str, Any]]:
        """FAISS(Dense) 상위 k 반환 (IP/코사인류)"""
        qv = self.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        D, I = self.index.search(qv, k)
        hits: List[Dict[str, Any]] = []
        for i, idx in enumerate(I[0]):
            if idx < 0:
                continue
            doc_id = self.meta["ids"][int(idx)]
            rec = dict(self.docs.get(doc_id, {}))
            rec["id"] = doc_id
            rec["score"] = float(D[0][i])
            hits.append(rec)
        return hits
    
    # ----------------- 결합기 -----------------

    @staticmethod
    def _rrf_fuse(
        bm25_hits: List[Dict[str, Any]],
        dense_hits: List[Dict[str, Any]],
        k: int,
        cand_factor: float,
        k0: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion.
        반환: [{id, score}] 형태, 길이 ≈ cand_factor*k
        """
        # id -> best rank
        ranks: Dict[str, float] = {}
        for r, h in enumerate(bm25_hits, start=1):
            ranks[h["id"]] = ranks.get(h["id"], float("inf"))
            ranks[h["id"]] = min(ranks[h["id"]], r)
        for r, h in enumerate(dense_hits, start=1):
            ranks[h["id"]] = ranks.get(h["id"], float("inf"))
            ranks[h["id"]] = min(ranks[h["id"]], r)

        fused = [{"id": did, "score": 1.0 / (k0 + rank)} for did, rank in ranks.items()]
        fused.sort(key=lambda x: x["score"], reverse=True)

        n = max(1, int(round(k * max(1.0, cand_factor))))
        return fused[:n]

    @staticmethod
    def _weighted_fuse(
        bm25_hits: List[Dict[str, Any]],
        dense_hits: List[Dict[str, Any]],
        k: int,
        alpha: float,
        cand_factor: float,
    ) -> List[Dict[str, Any]]:
        """
        가중합 결합: alpha * bm25_rank_score + (1-alpha) * dense_rank_score
        - 점수 스케일 차이를 줄이기 위해 rank 기반 1/(k0+rank) 변환 사용
        """
        k0 = 60.0

        def to_rank_score(hits: List[Dict[str, Any]]) -> Dict[str, float]:
            scores: Dict[str, float] = {}
            for r, h in enumerate(hits, start=1):
                scores[h["id"]] = 1.0 / (k0 + r)
            return scores

        bm25_s = to_rank_score(bm25_hits)
        dense_s = to_rank_score(dense_hits)

        fused: Dict[str, float] = {}
        for did in set(list(bm25_s.keys()) + list(dense_s.keys())):
            s = alpha * bm25_s.get(did, 0.0) + (1.0 - alpha) * dense_s.get(did, 0.0)
            fused[did] = fused.get(did, 0.0) + s

        out = [{"id": did, "score": sc} for did, sc in fused.items()]
        out.sort(key=lambda x: x["score"], reverse=True)
        n = max(1, int(round(k * max(1.0, cand_factor))))
        return out[:n]
    
    # ----------------- 후처리 -----------------

    def _merge(self, cand: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """후보 id 목록에 문서 필드 붙이기"""
        out: List[Dict[str, Any]] = []
        for c in cand:
            did = c["id"]
            base = self.docs.get(did, {})
            rec = {
                "id": did,
                "score": float(c.get("score", 0.0)),
                "law_name": base.get("law_name"),
                "clause_id": base.get("clause_id"),
                "title": base.get("title"),
                "text": base.get("text"),
            }
            out.append(rec)
        return out
    
    # ----------------- 공개 API -----------------

    def search(
        self,
        q: str,
        *,
        rerank: bool = True,
        k: int = FINAL_K,
        method: str = "rrf",           # "rrf" | "weighted"
        alpha: Optional[float] = None, # weighted일 때 bm25 비중(0~1)
        cand_factor: float = RERANK_CAND_FACTOR,
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색.
        1) BM25/Dense를 충분히 크게 가져온 뒤(cand_factor*k 이상),
        2) 결합(RRF 기본),
        3) (선택) CrossEncoder 리랭크 후 Top-k 반환.
        """
        # 후보 수 결정
        req_n = max(1, int(round(k * max(1.0, cand_factor))))
        bm25_k = max(BM25_K, req_n)
        dense_k = max(DENSE_K, req_n)

        # 1) 개별 검색
        bm25 = self._bm25(q, k=bm25_k)
        dense = self._dense(q, k=dense_k)

        # 2) 결합
        method = (method or "rrf").lower()
        if method.startswith("weight"):
            a = LAMBDA_BM25 if alpha is None else alpha
            try:
                a = float(a)
            except Exception:
                a = 0.5
            a = max(0.0, min(1.0, a))
            fused = self._weighted_fuse(bm25, dense, k=k, alpha=a, cand_factor=cand_factor)
        else:
            fused = self._rrf_fuse(bm25, dense, k=k, cand_factor=cand_factor)

        merged = self._merge(fused)

        # 3) 리랭크
        if rerank and self.reranker is not None:
            # CrossEncoder는 쿼리-문서 쌍 점수↑
            pairs = [(q, h["text"] or "") for h in merged]
            scores = self.reranker.predict(pairs)
            for h, s in zip(merged, scores):
                h["score"] = float(s)
            merged.sort(key=lambda x: x["score"], reverse=True)

        return merged[:k]
