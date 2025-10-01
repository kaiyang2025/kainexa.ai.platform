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
        """
        테스트에서 patch 대상으로 사용하는 배치 업서트 스텁.
        실제 구현에선 벡터/메타데이터를 묶어 upsert 하세요.
        """
        return True


class QdrantStore(VectorStore):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "default",
        url: Optional[str] = None,
    ):
        """
        host/port 또는 url 둘 다 지원 (테스트는 host/port/collection으로 호출)
        """
        self.host = host
        self.port = int(port)
        self.collection = collection
        self.url = url or f"http://{self.host}:{self.port}"

    async def connect(self) -> bool:
        # 실제 구현에선 헬스체크/컬렉션 생성 확인 등을 수행
        return True


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
    # fallback: negative L2 (값이 클수록 유사하게 보이도록 음수로 반환)
    diff = [a - b for a, b in zip(v1, v2)]
    return -math.sqrt(sum(d * d for d in diff))
