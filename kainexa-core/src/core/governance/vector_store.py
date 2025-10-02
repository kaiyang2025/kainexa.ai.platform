# -*- coding: utf-8 -*-
"""
In-Memory VectorStore (test-safe)
- 외부 라이브러리 없이 동작
- 간단한 TF(L2 정규화) + 코사인 유사도
- step_executors.RetrieveKnowledgeExecutor가 기대하는 인터페이스 준수:
    - 생성자: VectorStore()
    - async def search(query: str, k: int = 5, filter: Optional[dict] = None) -> List[dict]
      반환 항목 예: {"id": str, "text": str, "score": float, "metadata": dict}

추후 운영에서는 Qdrant/FAISS/Milvus 등으로 어댑터 교체 가능.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math
import re
import threading
import asyncio
import uuid

# --- 간단 토크나이저 (한글/영문/숫자) ---
_TOKEN = re.compile(r"[A-Za-z0-9가-힣]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN.findall(text)]


def _tf_vector(text: str) -> Dict[str, float]:
    toks = _tokenize(text)
    if not toks:
        return {}
    tf: Dict[str, float] = defaultdict(float)
    for t in toks:
        tf[t] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
    for k in list(tf.keys()):
        tf[k] /= norm
    return dict(tf)


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # 작은 쪽 기준으로 곱
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for k, v in a.items():
        bv = b.get(k)
        if bv:
            s += v * bv
    # cos 값은 [-1,1] 범위이나 여기서는 항상 0~1 근사
    return float(s)


@dataclass
class VectorItem:
    id: str
    text: str
    vector: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    테스트/개발용 인메모리 벡터스토어.

    주요 메소드
    ----------
    - upsert(documents, metadatas=None, ids=None) -> List[str]
    - async search(query: str, k: int = 5, filter: Optional[dict] = None) -> List[dict]
    - delete(ids: Optional[List[str]] = None, filter: Optional[dict] = None) -> int
    - count() -> int
    - reset() -> None
    """

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self._lock = threading.RLock()
        self._items: Dict[str, VectorItem] = {}

    # ---------------- Write APIs ----------------
    def upsert(
        self,
        documents: Iterable[str],
        metadatas: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
        ids: Optional[Iterable[Optional[str]]] = None,
    ) -> List[str]:
        """
        문서 삽입/갱신. 길이 불일치 시 ValueError.
        """
        docs = list(documents)
        metas = list(metadatas) if metadatas is not None else [None] * len(docs)
        id_list = list(ids) if ids is not None else [None] * len(docs)

        if not (len(docs) == len(metas) == len(id_list)):
            raise ValueError("documents/metadatas/ids length mismatch")

        new_ids: List[str] = []
        with self._lock:
            for i, text in enumerate(docs):
                _id = id_list[i] or str(uuid.uuid4())
                meta = metas[i] or {}
                vec = _tf_vector(text)
                self._items[_id] = VectorItem(id=_id, text=text, vector=vec, metadata=meta)
                new_ids.append(_id)
        return new_ids

    # ---------------- Read/Search APIs ----------------
    async def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        return_text: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        step_executors.RetrieveKnowledgeExecutor에서 호출됨.
        - 반환 리스트 항목은 최소 다음 키 포함:
          {"id": str, "score": float, "text": str, "metadata": dict}
        """
        # 비동기 시뮬레이션(이벤트 루프 양보) — 실제 백엔드 호출 대체
        await asyncio.sleep(0)

        qv = _tf_vector(query or "")
        filt = filter or {}
        scored: List[Tuple[str, float]] = []

        with self._lock:
            for _id, item in self._items.items():
                if filt and not _meta_match(item.metadata, filt):
                    continue
                score = _cosine(qv, item.vector)
                if score > 0.0:
                    scored.append((_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: max(1, int(k))]

        out: List[Dict[str, Any]] = []
        with self._lock:
            for _id, score in top:
                itm = self._items[_id]
                row = {
                    "id": _id,
                    "score": float(score),
                    "metadata": dict(itm.metadata),
                }
                if return_text:
                    row["text"] = itm.text
                out.append(row)
        return out

    # ---------------- Admin APIs ----------------
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        removed = 0
        with self._lock:
            if ids:
                for _id in ids:
                    if _id in self._items:
                        self._items.pop(_id)
                        removed += 1
                return removed

            if filter:
                keys = [k for k, it in self._items.items() if _meta_match(it.metadata, filter)]
                for k in keys:
                    self._items.pop(k)
                    removed += 1
                return removed

            # 전체 삭제
            removed = len(self._items)
            self._items.clear()
            return removed

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def reset(self) -> None:
        with self._lock:
            self._items.clear()


# ---------------- helpers ----------------
def _meta_match(meta: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """
    매우 단순한 메타데이터 매칭:
    - 값이 리스트/튜플/셋이면 포함 여부
    - 값이 dict면 부분 일치
    - 그 외는 == 비교
    """
    for k, v in flt.items():
        if k not in meta:
            return False
        mv = meta[k]
        if isinstance(v, (list, tuple, set)):
            if mv not in v:
                return False
        elif isinstance(v, dict):
            if not isinstance(mv, dict):
                return False
            for kk, vv in v.items():
                if mv.get(kk) != vv:
                    return False
        else:
            if mv != v:
                return False
    return True
