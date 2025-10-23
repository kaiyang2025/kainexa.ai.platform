## src/core/models/__init__.py
"""
Public API for src.core.models

- Re-export ORM entities so that:
    from src.core.models import User, Session, Conversation, Message
  works regardless of internal layout.
- Heavy / optional modules (LLM, tensor-parallel) are NOT imported here
  to avoid import-time side effects and missing-deps errors in tests.
"""

# --- ORM models (required) ---
from .orm_models import (
    Base,
    User,
    Session,
    Conversation,
    Message,
    KnowledgeDocument,
    KnowledgeChunk,
    AuditLog,
)

# --- Light factories/adapters (optional public surface) ---
# 필요할 때만 사용하도록 import를 지연시키는 래퍼를 제공합니다.
def get_model_factory():
    from .model_factory import ModelFactory  # local import to avoid heavy deps at import time
    return ModelFactory

__all__ = [
    # ORM
    "Base",
    "User",
    "Session",
    "Conversation",
    "Message",
    "KnowledgeDocument",
    "KnowledgeChunk",
    "AuditLog",
    # factory accessor
    "get_model_factory",
]
