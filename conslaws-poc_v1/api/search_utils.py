import json, pathlib, faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from opensearchpy import OpenSearch
from .config import (
    OPENSEARCH_URL, OPENSEARCH_INDEX,
    EMBED_MODEL, RERANK_MODEL,
    BM25_K, DENSE_K, FINAL_K, RERANK_CAND_FACTOR
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"
DOCS_PATH = INDEX_DIR / "docs.jsonl"
META_PATH = INDEX_DIR / "meta.json"

class Retriever:
    def __init__(self):
        self.client = OpenSearch(OPENSEARCH_URL, timeout=60)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
        self.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        self.docs = {}
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.docs[r["id"]] = r
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            try:
                self._reranker = CrossEncoder(RERANK_MODEL)
            except Exception:
                self._reranker = None
        return self._reranker

    def _bm25(self, q: str, k: int = BM25_K):
        body = {"size": k, "query": {"multi_match": {"query": q, "fields": ["title^2", "text^1"]}}}
        resp = self.client.search(index=OPENSEARCH_INDEX, body=body)
        hits = []
        for h in resp["hits"]["hits"]:
            src = h["_source"]
            hits.append({"id": h["_id"], "score": float(h["_score"]), **src})
        return hits

    def _dense(self, q: str, k: int = DENSE_K):
        qv = self.embedder.encode([q], normalize_embeddings=True)
        D, I = self.index.search(qv, k)
        hits = []
        for i, idx in enumerate(I[0]):
            if idx == -1:
                continue
            doc_id = self.meta["ids"][idx]
            src = self.docs.get(doc_id, {})
            hits.append({"id": doc_id, "score": float(D[0][i]), **src})
        return hits

    @staticmethod
    def _rrf_fuse(bm25_hits, dense_hits, k=FINAL_K, k_rrf=60, cand_factor=1.0):
        rank = {}
        for rank_list in [bm25_hits, dense_hits]:
            for i, h in enumerate(rank_list):
                rank[h["id"]] = rank.get(h["id"], 0.0) + 1.0/(k_rrf + i + 1)
        n = max(1, int(k * cand_factor))  # 리랭크 후보폭: cand_factor × k
        fused = [{"id": doc_id, "score": score}
                  for doc_id, score in sorted(rank.items(), key=lambda x: x[1], reverse=True)[:n]]
        return fused

    def _merge(self, fused):
        out = []
        for f in fused:
            d = self.docs.get(f["id"])
            if not d: 
                continue
            out.append({
                "id": f["id"],
                "score": f["score"],
                "law_name": d.get("law_name",""),
                "clause_id": d.get("clause_id",""),
                "title": d.get("title",""),
                "text": d.get("text",""),
            })
        return out

    def search(self, q: str, rerank: bool = True, k: int = FINAL_K,
               alpha: float = None,                    # None이면 RRF, 수치 주면 가중합
               cand_factor: float = RERANK_CAND_FACTOR # 리랭크 후보폭 (k의 배수)
               ):
        bm25 = self._bm25(q)
        dense = self._dense(q)
        # 결합 방식 선택
        if alpha is not None:
            try:
                a = float(alpha)
            except Exception:
                a = 0.5
            a = max(0.0, min(1.0, a))
            fused = self._weighted_fuse(bm25, dense, k=k, alpha=a, cand_factor=cand_factor)
        else:
            fused = self._rrf_fuse(bm25, dense, k=k, cand_factor=cand_factor)
        merged = self._merge(fused)
        if rerank and self.reranker is not None:
            pairs = [(q, h["text"]) for h in merged]
            scores = self.reranker.predict(pairs)
            for h, s in zip(merged, scores):
                h["score"] = float(s)
            merged = sorted(merged, key=lambda x: x["score"], reverse=True)
        return merged[:k]
