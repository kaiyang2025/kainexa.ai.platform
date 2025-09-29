#!/usr/bin/env python3
# kainexa-core/add_workflow_endpoint.py
# 기존 Core API에 워크플로우 실행 엔드포인트 추가

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Kainexa Core API with Workflow",
    version="1.0.0",
    description="Core API with Workflow Execution"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청/응답 모델
class WorkflowNode(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]

class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    type: Optional[str] = "default"

class WorkflowExecuteRequest(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = {}

class WorkflowExecuteResponse(BaseModel):
    execution_id: str
    status: str
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    timestamp: str

# 기존 엔드포인트
@app.get("/")
async def root():
    return {
        "name": "Kainexa AI Platform",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "chat": "/api/v1/chat",
            "documents": "/api/v1/documents",
            "scenarios": "/api/v1/scenarios",
            "health": "/api/v1/health/full",
            "workflow": "/api/v1/workflow/execute"  # 추가
        }
    }

@app.get("/api/v1/health/full")
async def health_full():
    return {
        "status": "healthy",
        "services": {
            "llm": {"status": "healthy", "model": "solar"},
            "rag": {"status": "healthy"},
            "database": {"status": "healthy"},
            "cache": {"status": "healthy"}
        },
        "timestamp": datetime.now().isoformat()
    }

# ✅ 워크플로우 실행 엔드포인트 추가
@app.post("/api/v1/workflow/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(request: WorkflowExecuteRequest):
    """워크플로우를 실행합니다."""
    
    print(f"\n{'='*50}")
    print(f"🚀 워크플로우 실행 요청")
    print(f"{'='*50}")
    print(f"노드 수: {len(request.nodes)}")
    print(f"연결 수: {len(request.edges)}")
    
    # 노드 정보 출력
    results = []
    for i, node in enumerate(request.nodes):
        node_id = node.get('id', f'node_{i}')
        node_type = node.get('type', 'unknown')
        node_data = node.get('data', {})
        
        print(f"\n📌 노드 {i+1}: {node_id}")
        print(f"  타입: {node_type}")
        print(f"  라벨: {node_data.get('label', 'N/A')}")
        
        # 노드 타입별 실행 시뮬레이션
        result = {
            "node_id": node_id,
            "type": node_type,
            "status": "executed",
            "output": None
        }
        
        if node_type == 'intent':
            result["output"] = {
                "intent": "greeting",
                "confidence": 0.95,
                "message": "의도 분류 완료: 인사"
            }
        elif node_type == 'llm':
            result["output"] = {
                "response": "안녕하세요! Kainexa AI입니다. 무엇을 도와드릴까요?",
                "model": node_data.get('config', {}).get('model', 'solar'),
                "tokens": 24
            }
        elif node_type == 'api':
            result["output"] = {
                "status_code": 200,
                "data": {"result": "API 호출 성공"},
                "url": node_data.get('config', {}).get('url', 'https://api.example.com')
            }
        elif node_type == 'condition':
            result["output"] = {
                "condition_met": True,
                "next_node": "node_2"
            }
        elif node_type == 'loop':
            result["output"] = {
                "iterations": 3,
                "completed": True
            }
        
        results.append(result)
    
    # 실행 완료
    print(f"\n✅ 워크플로우 실행 완료!")
    print(f"{'='*50}")
    
    return WorkflowExecuteResponse(
        execution_id=f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        status="completed",
        message=f"워크플로우가 성공적으로 실행되었습니다. {len(request.nodes)}개 노드 처리 완료.",
        results=results,
        timestamp=datetime.now().isoformat()
    )

# 채팅 엔드포인트
@app.post("/api/v1/chat")
async def chat(message: str = "안녕하세요"):
    """채팅 응답을 생성합니다."""
    return {
        "response": f"입력하신 '{message}'에 대한 AI 응답입니다.",
        "session_id": "test-session",
        "timestamp": datetime.now().isoformat()
    }

# 시나리오 엔드포인트들
@app.post("/api/v1/scenarios/production")
async def production_scenario(query: str = "생산 현황"):
    """생산 모니터링 시나리오"""
    return {
        "status": "success",
        "query": query,
        "data": {
            "total": {
                "planned": 1000,
                "actual": 950,
                "achievement_rate": 95.0,
                "defects": 10,
                "defect_rate": 1.05
            }
        },
        "report": f"{query}에 대한 분석: 생산 달성률 95%, 품질 양호",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Kainexa Core API 시작 (워크플로우 지원)")
    print("="*60)
    print("\n📡 접속 URL:")
    print("  - API: http://localhost:8000")
    print("  - API: http://192.168.1.215:8000")
    print("  - Docs: http://192.168.1.215:8000/docs")
    print("\n✅ 워크플로우 실행: POST /api/v1/workflow/execute")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)