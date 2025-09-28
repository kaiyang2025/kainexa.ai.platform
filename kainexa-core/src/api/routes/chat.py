# ============================================
# 1. src/api/routes/chat.py - 통합 채팅 API
# ============================================
"""src/api/routes/chat.py"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket
from typing import Dict, Any, List
import asyncio
import json
from datetime import datetime

from src.api.schemas.schemas import ChatRequest, ChatResponse
from src.auth.jwt_manager import get_current_user
from src.agents.chat_agent import ChatAgent
from src.core.cache import CacheManager
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# 전역 에이전트 인스턴스
chat_agent = ChatAgent()
cache_manager = CacheManager()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """통합 채팅 엔드포인트"""
    try:
        # 캐시 체크
        cache_key = f"chat:{current_user['user_id']}:{hash(request.message)}"
        cached = await cache_manager.get(cache_key)
        if cached:
            return ChatResponse(**cached)
        
        # 에이전트 실행
        response = await chat_agent.process(
            message=request.message,
            user_id=current_user['user_id'],
            session_id=request.session_id,
            context=request.context
        )
        
        # 캐시 저장
        await cache_manager.set(cache_key, response.dict(), ttl=300)
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket 실시간 채팅"""
    await websocket.accept()
    
    try:
        while True:
            # 메시지 수신
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 스트리밍 응답
            async for chunk in chat_agent.stream(
                message=message['text'],
                session_id=session_id
            ):
                await websocket.send_json({
                    "type": "chunk",
                    "data": chunk
                })
            
            # 완료 신호
            await websocket.send_json({
                "type": "complete",
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
