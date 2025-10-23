# ============================================
# 2. src/api/schemas/schemas.py - 통합 스키마
# ============================================
"""src/api/schemas/schemas.py"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class Role(str, Enum):
    USER = "user"
    AGENT = "agent"
    ADMIN = "admin"

class ChatRequest(BaseModel):
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID")
    conversation_id: Optional[str] = Field(None, description="대화 ID")   # ✅ 추가
    user_email: Optional[str] = Field(None, description="사용자 이메일")   # ✅ 추가
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    stream: bool = Field(False, description="스트리밍 여부")
    
class ChatResponse(BaseModel):
    response: str = Field(..., description="AI 응답")
    session_id: str
    intent: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: Role
    
    
class RagSearchRequest(BaseModel):
    query: str
    top_k: int = 5