# src/core/governance/vector_stores.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import math

class VectorStore:
    async def add_documents(self, docs: List[Dict[str, Any]]) -> bool:
        return True

    async def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        return []

    async def update_document(self, doc_id: str, new_content: str) -> bool:
        return True

    async def delete_document(self, doc_id: str) -> bool:
        return True

    async def batch_upsert(self, documents: List[Dict[str, Any]]) -> bool:
        """테스트에서 patch 대상으로 쓰는 스텁 — 실제 구현에선 벡터/메타 upsert"""
        return True

    async def add_batch(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        documents를 batch_size 단위로 끊어 batch_upsert 호출.
        하나라도 실패(False)면 False 반환.
        """
        n = len(documents or [])
        for i in range(0, n, max(1, batch_size)):
            chunk = documents[i:i + batch_size]
            ok = await self.batch_upsert(chunk)
            if not ok:
                return False
        return True


class QdrantStore(VectorStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "default",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        """
        host/port 또는 url 둘 다 지원 (테스트는 host/port/collection으로 호출)
        """
        self.host = host
        self.port = int(port)
        self.collection = collection
        self.url = url or f"http://{self.host}:{self.port}"
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self._client = None  # lazy init

    async def connect(self) -> bool:
        # 실제 구현에선 연결/헬스체크 수행
        try:
            import qdrant_client  # 테스트에서 여기 모듈을 patch 합니다.
            self._client = qdrant_client.QdrantClient(
                url=self.url,
                host=None if self.url else self.host,  # url 우선
                port=None if self.url else self.port,
                api_key=self.api_key,
                prefer_grpc=self.prefer_grpc,
            )
            return True
        except Exception:
            return False

    async def test_connection(self) -> bool:
        """
        client.get_collection(self.collection)을 호출해 연결 확인.
        테스트에서 qdrant_client.QdrantClient가 patch되므로, 내부 import를 유지해야 함.
        """
        if self._client is None:
            await self.connect()
        if self._client is None:
            return False
        try:
            _ = self._client.get_collection(self.collection)
            return True
        except Exception:
            return False


def compute_similarity(v1: List[float], v2: List[float], metric: str = "cosine") -> float:
    if metric == "cosine":
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1)) or 0.0
        n2 = math.sqrt(sum(b * b for b in v2)) or 0.0
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)
    if metric == "dot":
        return sum(a * b for a, b in zip(v1, v2))
    # fallback: negative L2 (값이 클수록 유사하도록 음수 반환)
    diff = [a - b for a, b in zip(v1, v2)]
    return -math.sqrt(sum(d * d for d in diff))
