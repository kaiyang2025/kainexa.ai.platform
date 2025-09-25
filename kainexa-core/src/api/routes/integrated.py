# src/api/routes/integrated.py 생성
"""
통합 API 엔드포인트
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import get_db
from src.core.session_manager import SessionManager, ConversationManager
from src.models.solar_llm import SolarLLM
from src.governance.rag_pipeline import RAGGovernance, DocumentMetadata, AccessLevel
from src.scenarios.production_monitoring import ProductionMonitoringAgent
from src.scenarios.predictive_maintenance import PredictiveMaintenanceAgent
from src.scenarios.quality_control import QualityControlAgent


router = APIRouter(prefix="/api/v1", tags=["integrated"])

# Pydantic 모델
class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    # pydantic v2에선 mutable default도 안전하지만, 의미상 factory가 더 명시적
    context: Dict[str, Any] = Field(default_factory=dict)

class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    chunks: int
    status: str

# 전역 인스턴스
llm = SolarLLM()
rag = RAGGovernance()

@router.post("/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """로그인"""
    
    # 간단한 인증 (실제는 암호화 필요)
    from sqlalchemy import select
    from src.core.models import User
        
    # 사용자 조회 (email 기반)
    query = select(User).where(User.email == request.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # 세션 생성
    session_manager = SessionManager(db)
    session_data = await session_manager.create_session(
        user_id=str(user.id),
        ip_address="127.0.0.1"
    )
    
    return {
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role,
        "session": session_data
    }

@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """채팅 엔드포인트"""
    
    # 대화 관리자
    conv_manager = ConversationManager(db)
    
    # 대화 생성 또는 조회
    if request.conversation_id:
        conversation_id = request.conversation_id
    else:
        # 세션/유저 정보가 없으면 ConversationManager가 게스트/세션을 생성하도록 맡깁니다.
        conversation = await conv_manager.create_conversation(
            # 제목은 한국시간(KST) 기준으로 보기 좋게
            title=f"대화 {datetime.now(ZoneInfo('Asia/Seoul')):%Y-%m-%d %H:%M}",
            context=request.context
        )
        conversation_id = str(conversation.id)
    
    # 사용자 메시지 저장
    await conv_manager.add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message
    )
    
    # RAG 검색
    rag_results = await rag.retrieve(
        query=request.message,
        k=3,
        user_access_level=AccessLevel.INTERNAL
    )
    
    # LLM 응답 생성
    llm.load()
    
    # 프롬프트 구성
    prompt = f"사용자: {request.message}\n\n"
    
    if rag_results:
        prompt += "관련 정보:\n"
        for result in rag_results[:2]:
            prompt += f"- {result['text'][:200]}...\n"
        prompt += "\n"
    
    prompt += "응답:"
    
    response_text = llm.generate(
        prompt,
        max_new_tokens=300,
        temperature=0.7
    )
    
    # 어시스턴트 메시지 저장
    await conv_manager.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=response_text
    )
    
    return {
        "conversation_id": conversation_id,
        "response": response_text,
        "sources": [r.get('source', '') for r in rag_results] if rag_results else [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """문서 업로드"""
    
    # 파일 읽기
    content = await file.read()
    text_content = content.decode('utf-8')
    
    # 메타데이터 생성
    metadata = DocumentMetadata(
        doc_id=f"upload_{datetime.now():%Y%m%d_%H%M%S}",
        title=file.filename,
        source=f"upload/{file.filename}",        
        created_at=datetime.now(timezone.utc),
        access_level=AccessLevel.INTERNAL,
        tags=["upload"],
        language="ko"
    )
    
    # RAG에 추가
    success = await rag.add_document(text_content, metadata)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process document")
    
    return DocumentUploadResponse(
        document_id=metadata.doc_id,
        title=metadata.title,
        chunks=len(text_content) // 500,  # 대략적인 청크 수
        status="success"
    )

@router.get("/documents/search")
async def search_documents(
    query: str,
    limit: int = 5,
    db: AsyncSession = Depends(get_db)
):
    """문서 검색"""
    
    results = await rag.retrieve(
        query=query,
        k=limit,
        user_access_level=AccessLevel.INTERNAL
    )
    
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }

@router.post("/scenarios/production")
async def run_production_scenario(
    query: str = "생산 현황 보고해줘",
    db: AsyncSession = Depends(get_db)
):
    """생산 모니터링 시나리오"""
    
    agent = ProductionMonitoringAgent()
    result = await agent.analyze_production(query)
    
    return result

@router.post("/scenarios/maintenance")
async def run_maintenance_scenario(
    equipment_id: str = "CNC_007",
    db: AsyncSession = Depends(get_db)
):
    """예측적 유지보수 시나리오"""
    
    agent = PredictiveMaintenanceAgent()
    result = await agent.predict_failure(equipment_id)
    
    return result

@router.post("/scenarios/quality")
async def run_quality_scenario(
    db: AsyncSession = Depends(get_db)
):
    """품질 관리 시나리오"""
    
    agent = QualityControlAgent()
    result = await agent.analyze_quality()
    
    return result

@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """대화 기록 조회"""
    
    conv_manager = ConversationManager(db)
    messages = await conv_manager.get_conversation_history(
        conversation_id,
        limit=limit
    )
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {
                "id": str(msg.id),
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat()
            }
            for msg in messages
        ]
    }

@router.get("/health/full")
async def full_health_check():
    """전체 시스템 헬스체크"""
    
    health_status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {}
    }
    
    # LLM 체크
    try:
        llm.load()
        memory = llm.get_memory_usage()
        health_status["services"]["llm"] = {
            "status": "healthy",
            "memory": memory
        }
    except Exception as e:
        health_status["services"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # RAG 체크
    try:
        report = rag.get_governance_report()
        health_status["services"]["rag"] = {
            "status": "healthy",
            "collections": report.get("collection_stats", {})
        }
    except Exception as e:
        health_status["services"]["rag"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # GPU 체크
    import torch
    health_status["services"]["gpu"] = {
        "available": torch.cuda.is_available(),
        "count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return health_status
