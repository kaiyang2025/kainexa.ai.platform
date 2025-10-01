# src/api/routes/ws.py
from fastapi import APIRouter, WebSocket
import uuid
import asyncio

router = APIRouter()

@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    session_id = None
    while True:
        data = await ws.receive_json()
        t = data.get("type")
        if t == "init":
            session_id = f"ws_{uuid.uuid4().hex[:8]}"
            await ws.send_json({"type": "init_success", "session_id": session_id})
        elif t == "message":
            content = data.get("content", "")
            await ws.send_json({"type": "response", "content": f"echo: {content}", "metadata": {"session_id": session_id}})
        else:
            break

@router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_json()
        if data.get("type") == "stream_start":
            # 3개의 청크로 스트리밍
            for part in ["chunk-1 ", "chunk-2 ", "chunk-3"]:
                await ws.send_json({"type": "stream_chunk", "content": part})
                await asyncio.sleep(0.01)
            await ws.send_json({"type": "stream_end"})
