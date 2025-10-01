from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])

class CreateConversationRequest(BaseModel):
    user_id: str | None = None

class CreateConversationResponse(BaseModel):
    conversation_id: str

class MessageRequest(BaseModel):
    role: str  # "user"/"assistant"
    content: str

class MessageResponse(BaseModel):
    conversation_id: str
    reply: str

@router.post("", response_model=CreateConversationResponse)
async def create_conversation(_: CreateConversationRequest):
    return CreateConversationResponse(conversation_id=str(uuid.uuid4()))

@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def add_message(conversation_id: str, req: MessageRequest):
    if not req.content:
        raise HTTPException(status_code=400, detail="content is required")
    # 단순 에코/스텁
    return MessageResponse(conversation_id=conversation_id, reply=f"echo: {req.content}")
