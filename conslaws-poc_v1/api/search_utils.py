# -*- coding: utf-8 -*-
import json
import pathlib
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from collections import defaultdict

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import (
    OPENSEARCH_URL, OPENSEARCH_INDEX,
    EMBED_MODEL, RERANK_MODEL,    
    BM25_K, DENSE_K, FINAL_K,
    RERANK_CAND_FACTOR,
    LAMBDA_BM25, LAMBDA_DENSE,
    SEARCH_METHOD, USE_RERANK,
)

# 경로 상수
ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index"
DOCS_PATH = INDEX_DIR / "docs.jsonl"
CHROMA_DIR = INDEX_DIR / "chroma"
MAPPING_PATH = INDEX_DIR / "law_enfor_mapping.json"

class Retriever:    
    """
    하이브리드 검색기
    - OpenSearch(BM25) + ChromaDB(Dense)
    - 문맥 확장: 법령 <-> 시행령 자동 연결
    - 결합: RRF(기본) 또는 가중합
    - 리랭크: CrossEncoder(옵션)
    """
    def __init__(self,
        opensearch_url: Optional[str] = None,
        index_name: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        chroma_dir: Optional[str] = None, # FAISS_DIR 대신 CHROMA_DIR 사용
        reranker_name: Optional[str] = None,
    ):
        # ---- 설정 ----
        self.opensearch_url = opensearch_url or OPENSEARCH_URL
        self.index_name = index_name or OPENSEARCH_INDEX
        self.embed_model_name = embed_model_name or EMBED_MODEL
        self.chroma_path = chroma_dir or CHROMA_DIR
        
        # ---- 문서 데이터 로드 (Lookup용) ----
        # 검색 후 원본 텍스트나 메타데이터를 빠르게 찾기 위해 메모리에 로드
        if not DOCS_PATH.exists():
            raise FileNotFoundError(f"Docs file not found: {DOCS_PATH}")

        self.docs: Dict[str, Dict[str, Any]] = {}
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                self.docs[r["id"]] = r

        # ---- 매핑 및 룩업 테이블 구축 (확장용) ----
        self.enfor_to_law = {}
        self.law_to_enfor = defaultdict(list)
        
        if MAPPING_PATH.exists():
            with open(MAPPING_PATH, "r", encoding="utf-8") as f:
                self.enfor_to_law = json.load(f)
            for enfor_key, law_list in self.enfor_to_law.items():
                for law_key in law_list:
                    self.law_to_enfor[law_key].append(enfor_key)
        
        # 조문 단위 그룹핑
        self.article_lookup = defaultdict(list)
        for doc_id, doc in self.docs.items():
            try:
                base_key = doc_id.split("-")[0] if "-" in doc_id else doc_id
                self.article_lookup[base_key].append(doc)
            except:
                pass
        
        # ---- OpenSearch (BM25) ----
        self.client = OpenSearch(self.opensearch_url, timeout=60)
        
        # ---- ChromaDB (Dense) ----  
        self.embedder = SentenceTransformer(self.embed_model_name)
        
        if not pathlib.Path(self.chroma_path).exists():
             raise FileNotFoundError(f"ChromaDB dir not found: {self.chroma_path}")

        print(f"[Info] Loading ChromaDB from {self.chroma_path}...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path), 
            settings=Settings(allow_reset=True)
        )
        self.collection = self.chroma_client.get_collection("laws_bge_m3_v2")

        # ---- 리랭커 (지연 로드) ----
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_name = reranker_name or RERANK_MODEL
    
    @property
    def reranker(self) -> Optional[CrossEncoder]:
        if self._reranker is None and self._reranker_name:
            try:
                self._reranker = CrossEncoder(self._reranker_name)
            except Exception:
                self._reranker = None
        return self._reranker

    def _bm25(self, q: str, k: int) -> List[Dict[str, Any]]:
        """OpenSearch BM25 검색"""
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
            doc_id = h.get("_id")
            # score와 기본 정보만 가져옴 (상세 내용은 self.docs에서 merge)
            rec = {
                "id": doc_id,
                "score": float(h.get("_score", 0.0)),
            }
            # docs에 있는 정보 채우기
            if doc_id in self.docs:
                rec.update(self.docs[doc_id])
            hits.append(rec)
        return hits

    def _dense(self, q: str, k: int) -> List[Dict[str, Any]]:
        """ChromaDB Dense 검색"""
        # 1. 쿼리 임베딩
        qv = self.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        
        # 2. Chroma 검색
        results = self.collection.query(
            query_embeddings=qv.tolist(),
            n_results=k,
            include=["documents", "distances"] # 메타데이터는 docs.jsonl에서 가져옴
        )
        
        hits: List[Dict[str, Any]] = []
        if results['ids']:
            ids = results['ids'][0]
            dists = results['distances'][0]
            
            for i, doc_id in enumerate(ids):
                # Cosine Distance -> Similarity 변환 (1 - distance)
                # ChromaDB의 'cosine' space는 1 - similarity를 반환함
                score = 1.0 - dists[i]
                
                rec = {
                    "id": doc_id,
                    "score": float(score)
                }
                # 상세 정보 채우기
                if doc_id in self.docs:
                    rec.update(self.docs[doc_id])
                hits.append(rec)
        return hits
    
    def _expand_results(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과의 연관 법령/시행령을 찾아 강제로 추가 (Expansion)"""
        expanded = []
        seen_ids = set(h["id"] for h in hits)
        
        for doc in hits:
            expanded.append(doc)
            
            # ID 파싱
            current_id = doc["id"]
            if "-" in current_id:
                article_key = current_id.split("-")[0]
            else:
                article_key = current_id
                
            targets = []
            if article_key in self.enfor_to_law:
                targets.extend(self.enfor_to_law[article_key])
            if article_key in self.law_to_enfor:
                targets.extend(self.law_to_enfor[article_key])
                
            for t_key in targets:
                # 해당 조문의 모든 항 가져오기
                for related_doc in self.article_lookup.get(t_key, []):
                    if related_doc["id"] not in seen_ids:
                        new_doc = related_doc.copy()
                        new_doc["score"] = doc["score"] * 0.95 # 부모 점수의 95%
                        new_doc["is_expanded"] = True
                        new_doc["parent_id"] = current_id
                        
                        expanded.append(new_doc)
                        seen_ids.add(related_doc["id"])
                        
        return expanded

    # --- 결합 로직 (RRF / Weighted) ---
    @staticmethod
    def _rrf_fuse(bm25_hits, dense_hits, k, cand_factor, k0=60):
        ranks = {}
        for r, h in enumerate(bm25_hits, start=1):
            ranks[h["id"]] = min(ranks.get(h["id"], float("inf")), r)
        for r, h in enumerate(dense_hits, start=1):
            ranks[h["id"]] = min(ranks.get(h["id"], float("inf")), r)

        fused = [{"id": did, "score": 1.0 / (k0 + rank)} for did, rank in ranks.items()]
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        n = max(1, int(round(k * max(1.0, cand_factor))))
        return fused[:n]

    @staticmethod
    def _weighted_fuse(bm25_hits, dense_hits, k, alpha, cand_factor):
        k0 = 60.0
        def to_rank_score(hits):
            return {h["id"]: 1.0/(k0 + i + 1) for i, h in enumerate(hits)}

        bm25_s = to_rank_score(bm25_hits)
        dense_s = to_rank_score(dense_hits)
        
        fused = {}
        for did in set(list(bm25_s.keys()) + list(dense_s.keys())):
            s = alpha * bm25_s.get(did, 0.0) + (1.0 - alpha) * dense_s.get(did, 0.0)
            fused[did] = fused.get(did, 0.0) + s
            
        out = [{"id": did, "score": sc} for did, sc in fused.items()]
        out.sort(key=lambda x: x["score"], reverse=True)
        n = max(1, int(round(k * max(1.0, cand_factor))))
        return out[:n]

    def _merge(self, cand, bm25_map=None, dense_map=None):
        out = []
        for c in cand:
            did = c["id"]
            # self.docs에서 원본 데이터 조회
            base = self.docs.get(did, {})
            rec = {
                "id": did,
                "score": float(c.get("score", 0.0)),
                "law_name": base.get("law_name"),
                "clause_id": base.get("clause_id"),
                "title": base.get("title"),
                "text": base.get("text"),
            }
            if bm25_map: rec["bm25_score"] = bm25_map.get(did)
            if dense_map: rec["dense_cosine"] = dense_map.get(did)
            out.append(rec)
        return out

    # --- 메인 검색 함수 ---
    def search(
        self,
        q: str,
        *,
        rerank: bool = True,
        k: int = FINAL_K,
        method: Optional[str] = None,
        alpha: Optional[float] = None,
        cand_factor: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        
        if cand_factor is None: cand_factor = RERANK_CAND_FACTOR
        
        # 확장 고려하여 2배수로 검색
        req_n = max(1, int(round(k * max(1.0, cand_factor))))
        bm25_k = max(BM25_K, req_n * 2)
        dense_k = max(DENSE_K, req_n * 2)

        # 1. 개별 검색
        bm25 = self._bm25(q, k=bm25_k)
        dense = self._dense(q, k=dense_k)
        
        # 2. 확장 (Expansion)
        bm25 = self._expand_results(bm25)
        dense = self._expand_results(dense)

        # 원점수 기록
        bm25_map = {h["id"]: h.get("score", 0.0) for h in bm25}
        dense_map = {h["id"]: h.get("score", 0.0) for h in dense}

        # 3. 결합
        method = (method or SEARCH_METHOD or "rrf").lower()
        if method.startswith("weight"):
            a = alpha if alpha is not None else LAMBDA_BM25
            fused = self._weighted_fuse(bm25, dense, k, a, cand_factor)
        else:
            fused = self._rrf_fuse(bm25, dense, k, cand_factor)
            
        # 4. 병합
        merged = self._merge(fused, bm25_map, dense_map)

        # 5. 리랭크
        if rerank and USE_RERANK and self.reranker:
            pairs = [(q, h["text"] or "") for h in merged]
            scores = self.reranker.predict(pairs)
            for h, s in zip(merged, scores):
                h["score"] = float(s)
            merged.sort(key=lambda x: x["score"], reverse=True)
            
        return merged[:k]