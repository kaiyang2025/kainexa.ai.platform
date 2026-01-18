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
    하이브리드 검색기 (주피터 노트북 V2.0 로직 완전 동기화)
    - OpenSearch(BM25) + ChromaDB(Dense)
    - 문맥 확장(Expansion): 법령 <-> 시행령 자동 연결 (부모 뒤에 자식 강제 배치)
    - 결합: RRF
    - 리랭크: CrossEncoder + Family Grouping (부모-자식 정렬 보장)
    """
    def __init__(self,
        opensearch_url: Optional[str] = None,
        index_name: Optional[str] = None,
        embed_model_name: Optional[str] = None,
        chroma_dir: Optional[str] = None, 
        reranker_name: Optional[str] = None,
    ):
        # ---- 설정 ----
        self.opensearch_url = opensearch_url or OPENSEARCH_URL
        self.index_name = index_name or OPENSEARCH_INDEX
        self.embed_model_name = embed_model_name or EMBED_MODEL
        self.chroma_path = chroma_dir or CHROMA_DIR
        
        # ---- 1. 문서 데이터 로드 (Lookup용) ----
        if not DOCS_PATH.exists():
            raise FileNotFoundError(f"Docs file not found: {DOCS_PATH}")

        self.docs: Dict[str, Dict[str, Any]] = {}
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                self.docs[r["id"]] = r

        # ---- 2. 매핑 및 룩업 테이블 구축 (확장용) ----
        self.enfor_to_law = {}
        self.law_to_enfor = defaultdict(list)
        
        if MAPPING_PATH.exists():
            with open(MAPPING_PATH, "r", encoding="utf-8") as f:
                self.enfor_to_law = json.load(f)
            # 역방향 인덱스 생성 (법 -> 시행령)
            for enfor_key, law_list in self.enfor_to_law.items():
                for law_key in law_list:
                    self.law_to_enfor[law_key].append(enfor_key)
        
        # 조문 단위 그룹핑 (ID 검색용)
        self.article_lookup = defaultdict(list)
        for doc_id, doc in self.docs.items():
            try:
                # "제10조-①" -> "제10조" 키로 그룹핑
                base_key = doc_id.split("-")[0] if "-" in doc_id else doc_id
                self.article_lookup[base_key].append(doc)
            except:
                pass
        
        # ---- 3. OpenSearch (BM25) ----
        self.client = OpenSearch(self.opensearch_url, timeout=60)
        
        # ---- 4. ChromaDB (Dense) ----  
        self.embedder = SentenceTransformer(self.embed_model_name)
        
        if not pathlib.Path(self.chroma_path).exists():
             print(f"[Warn] ChromaDB dir not found: {self.chroma_path}")
        
        print(f"[Info] Loading ChromaDB from {self.chroma_path}...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path), 
            settings=Settings(allow_reset=True)
        )
        self.collection = self.chroma_client.get_collection("laws_bge_m3_v2")

        # ---- 5. 리랭커 (지연 로드) ----
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_name = reranker_name or RERANK_MODEL
    
    @property
    def reranker(self) -> Optional[CrossEncoder]:
        if self._reranker is None and self._reranker_name:
            try:
                # 노트북과 동일한 설정 (max_length=512)
                self._reranker = CrossEncoder(self._reranker_name, max_length=512)
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
            rec = {
                "id": doc_id,
                "score": float(h.get("_score", 0.0)),
            }
            if doc_id in self.docs:
                rec.update(self.docs[doc_id])
            hits.append(rec)
        return hits

    def _dense(self, q: str, k: int) -> List[Dict[str, Any]]:
        """ChromaDB Dense 검색"""
        qv = self.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        results = self.collection.query(
            query_embeddings=qv.tolist(),
            n_results=k,
            include=["documents", "distances"] 
        )
        hits: List[Dict[str, Any]] = []
        if results['ids']:
            ids = results['ids'][0]
            dists = results['distances'][0]
            for i, doc_id in enumerate(ids):
                # 1 - distance = similarity
                score = 1.0 - dists[i]
                rec = {
                    "id": doc_id,
                    "score": float(score)
                }
                if doc_id in self.docs:
                    rec.update(self.docs[doc_id])
                hits.append(rec)
        return hits
    
    def _expand_results(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [노트북 로직 이식] 검색 결과 확장
        - 중요: processed_ids를 빈 집합으로 시작해야, 하위권에 있는 자식 문서도 부모 뒤로 끌어올림(Re-positioning)
        """
        expanded = []
        processed_ids = set() # [수정됨] 초기화: 빈 집합 (hits의 ID를 미리 넣지 않음!)
        
        for doc in hits:
            # 1. 이미 처리된 문서(상위 순위에서 추가된 경우)는 건너뜀
            if doc["id"] in processed_ids:
                continue
            
            expanded.append(doc)
            processed_ids.add(doc["id"])
            
            # 2. 연관 문서(가족) 찾기
            current_id = doc["id"]
            if "-" in current_id:
                article_key = current_id.split("-")[0]
            else:
                article_key = current_id
                
            targets = []
            # (A) 시행령 -> 법령
            if article_key in self.enfor_to_law:
                targets.extend(self.enfor_to_law[article_key])
            # (B) 법령 -> 시행령
            if article_key in self.law_to_enfor:
                targets.extend(self.law_to_enfor[article_key])
                
            # 3. 자식 문서 강제 삽입 (Injection)
            for t_key in targets:
                for related_doc in self.article_lookup.get(t_key, []):
                    # 이미 처리된 문서인지 확인 (여기서 하위권에 있던 문서는 아직 처리 안 된 상태)
                    if related_doc["id"] in processed_ids:
                        continue
                        
                    new_doc = related_doc.copy()
                    
                    # 부모가 법령이면, 자식(시행령)에게 부모 ID 부여
                    if article_key in self.law_to_enfor:
                        new_doc["parent_id"] = current_id
                    
                    # 점수 보정 (부모 점수의 95% -> 부모 바로 뒤에 위치하도록 유도)
                    base_score = doc.get("score", 0.0)
                    if base_score == 0.0: base_score = 0.5
                    new_doc["score"] = base_score * 0.95 
                    new_doc["is_expanded"] = True
                    
                    # ★ 리스트에 추가 (부모 바로 뒤)
                    expanded.append(new_doc)
                    processed_ids.add(related_doc["id"])
                        
        return expanded

    def _apply_family_grouping(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [노트북 로직 이식] 리랭킹 후 부모-자식 그룹핑
        - 리랭킹 점수가 낮아 흩어진 자식(시행령)을 부모(법령) 뒤로 다시 모음
        """
        final_res = []
        seen_ids = set()
        
        # (A) 자식 문서 맵핑 (parent_id -> [child_doc1, child_doc2...])
        children_map = defaultdict(list)
        for doc in candidates:
            if doc.get("is_expanded") and "parent_id" in doc:
                children_map[doc["parent_id"]].append(doc)

        # (B) 순회 및 배치
        for doc in candidates:
            if doc["id"] in seen_ids: continue

            # 나는 자식인데, 내 부모가 후보군 어딘가에 있다면? -> 대기(Skip)
            # (부모가 출력될 때 같이 출력되기 위함)
            is_child = doc.get("is_expanded") and "parent_id" in doc
            parent_exists_in_candidates = False
            
            if is_child:
                for d in candidates:
                    if d["id"] == doc["parent_id"]:
                        parent_exists_in_candidates = True
                        break
            
            # 부모가 후보군에 살아있다면, 나는 부모 나올 때까지 기다림
            if is_child and parent_exists_in_candidates:
                continue

            # 문서 추가
            final_res.append(doc)
            seen_ids.add(doc["id"])
            
            # 만약 내가 부모라면? 내 자식들을 바로 뒤에 줄세움
            if doc["id"] in children_map:
                my_children = children_map[doc["id"]]
                # 자식끼리는 점수순 정렬
                my_children.sort(key=lambda x: -x.get("score", 0))
                
                for child in my_children:
                    if child["id"] not in seen_ids:
                        # 점수를 부모보다 아주 조금 낮게 조정 (순서 보장용)
                        child["score"] = doc.get("score", 0) - 0.0001
                        final_res.append(child)
                        seen_ids.add(child["id"])
        
        return final_res

    # --- 결합 로직 (RRF) ---
    @staticmethod
    def _rrf_fuse(bm25_hits, dense_hits, k, cand_factor, k0=60):
        ranks = {}
        for r, h in enumerate(bm25_hits, start=1):
            ranks[h["id"]] = min(ranks.get(h["id"], float("inf")), r)
        for r, h in enumerate(dense_hits, start=1):
            ranks[h["id"]] = min(ranks.get(h["id"], float("inf")), r)

        fused = [{"id": did, "score": 1.0 / (k0 + rank)} for did, rank in ranks.items()]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused # 전체 반환

    def _merge(self, cand, bm25_map=None, dense_map=None):
        out = []
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
                "is_expanded": c.get("is_expanded", False),
                "parent_id": c.get("parent_id"), 
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
        
        # [수정] 확장(Expansion)을 위해 1차 후보군을 넉넉히 가져옵니다 (5배수 권장)
        req_n = max(1, int(round(k * max(1.0, cand_factor))))
        bm25_k = max(BM25_K, req_n * 5) 
        dense_k = max(DENSE_K, req_n * 5)
        
        # 1. 개별 검색
        bm25_res = self._bm25(q, k=bm25_k)
        dense_res = self._dense(q, k=dense_k)
        
        # 2. 문맥 확장 (Expansion) - 여기서 부모-자식 연결 및 순위 부스팅 발생
        bm25 = self._expand_results(bm25_res)
        dense = self._expand_results(dense_res)

        # 원점수 기록
        bm25_map = {h["id"]: h.get("score", 0.0) for h in bm25}
        dense_map = {h["id"]: h.get("score", 0.0) for h in dense}

        # 3. RRF 결합
        fused = self._rrf_fuse(bm25, dense, k, cand_factor)
        
        # 4. 상세 정보 병합
        merged = self._merge(fused, bm25_map, dense_map)

        # 5. 리랭크 (CrossEncoder) & 그룹핑
        if rerank and USE_RERANK and self.reranker:
            # [노트북 핵심] 리랭킹 후보군을 150개로 고정하여 자식 문서 누락 방지
            candidates = merged[:150] 
            
            pairs = [(q, h["text"] or "") for h in candidates]
            scores = self.reranker.predict(pairs)
            for h, s in zip(candidates, scores):
                h["score"] = float(s)
            
            # 점수순 1차 정렬
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # [노트북 핵심] 부모-자식 그룹핑 (최종 순서 정리)
            final_res = self._apply_family_grouping(candidates)
            
            return final_res[:k]
        
        else:
            # 리랭크 안 할 경우 그냥 RRF 점수순
            merged.sort(key=lambda x: x["score"], reverse=True)
            return merged[:k]