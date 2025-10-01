from __future__ import annotations
from typing import Any, Dict, List
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

class QdrantStore(VectorStore):
    def __init__(self, url: str = "http://localhost:6333", collection: str = "default"):
        self.url = url
        self.collection = collection
    async def connect(self) -> bool:
        return True

def compute_similarity(v1: List[float], v2: List[float], metric: str = "cosine") -> float:
    if metric == "cosine":
        dot = sum(a*b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a*a for a in v1)) or 0.0
        n2 = math.sqrt(sum(b*b for b in v2)) or 0.0
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return dot / (n1 * n2)
    if metric == "dot":
        return sum(a*b for a, b in zip(v1, v2))
    # fallback: negative L2 (값이 클수록 유사하게 보이도록 음수)
    diff = [a-b for a, b in zip(v1, v2)]
    return -math.sqrt(sum(d*d for d in diff))
