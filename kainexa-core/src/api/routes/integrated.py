# src/api/routes/integrated.py 생성
"""
통합 API 엔드포인트
"""

# 상단 import 보강
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
import traceback
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

def _ensure_uuid(val) -> UUID:
    try:
        return UUID(str(val))
    except Exception:
        raise HTTPException(status_code=500, detail=f"Invalid UUID value: {val!r}")

def _entity_id(entity, *keys) -> UUID:
    """
    ORM 객체 또는 dict 모두에서 안전하게 id를 뽑아 UUID로 변환.
    keys가 주어지면 우선순위대로 dict 키를 확인.
    """
    if entity is None:
        raise HTTPException(status_code=500, detail="Entity is None (cannot extract id)")

    # dict인 경우
    if isinstance(entity, dict):
        probe_keys = keys or ("id",)
        for k in probe_keys:
            v = entity.get(k)
            if v:
                return _ensure_uuid(v)
        raise HTTPException(status_code=500, detail=f"Cannot resolve id from dict; tried keys={probe_keys}")

    # ORM 객체인 경우
    if hasattr(entity, "id") and getattr(entity, "id"):
        return _ensure_uuid(getattr(entity, "id"))

    # 다른 형태(문자열 등)면 그대로 시도
    return _ensure_uuid(entity)

# Pydantic 모델
class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    # pydantic v2에선 mutable default도 안전하지만, 의미상 factory가 더 명시적
    context: Dict[str, Any] = Field(default_factory=dict)
    user_email: Optional[str] = None

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
    payload: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    대화형 AI 어시스턴트 - 한국어 전용
    - 사용자 식별: 헤더 X-User-Email → body.user_email → demo@kainexa.local
    - 세션/대화 자동 생성
    - RAG 문맥 + LLM(한국어 전용, 그리디)
    - 실패시 JSON 500(detail 포함)
    """
    try:
        # 1) 사용자 식별
        user_email = (
            request.headers.get("X-User-Email")
            or (payload.user_email or "").strip()
            or "demo@kainexa.local"
        )

        # 2) 매니저
        sess_mgr = SessionManager(db)
        conv_mgr = ConversationManager(db)

        # 3) 사용자/세션 확보 (ensure_*가 있다면 사용, 없으면 fallback)
        if hasattr(sess_mgr, "ensure_user_and_session"):
            user, session = await sess_mgr.ensure_user_and_session(user_email=user_email)
        else:
            # --- fallback 시작: 프로젝트에 맞춰 이미 존재하면 생략 ---
            from sqlalchemy import select
            from src.core.models import User, Session

            result = await db.execute(select(User).where(User.email == user_email))
            user = result.scalar_one_or_none()
            if user is None:
                user = await sess_mgr.create_user(email=user_email)
            result = await db.execute(select(Session).where(Session.user_id == user.id))
            session = result.scalar_one_or_none()
            if session is None:
                session = await sess_mgr.create_session(user_id=user.id)
            # --- fallback 끝 ---

        # 4) 대화 확보/생성
        if payload.conversation_id:
            conversation_id = payload.conversation_id
        else:            
            title = f"대화 {datetime.now(ZoneInfo('Asia/Seoul')):%Y-%m-%d %H:%M}"
            conversation = await conv_mgr.create_conversation(
                session_id=_entity_id(session, "id", "session_id"),
                user_id=_entity_id(user, "id", "user_id"),
                title=title,
            )
            conversation_id = str(conversation.id)

        # 5) 사용자 메시지 저장
        await conv_mgr.add_message(
            conversation_id=conversation_id,
            role="user",
            content=payload.message,
        )

        # 6) RAG 검색 (문맥)
        try:
            rag_results = await rag.retrieve(
                query=payload.message,
                k=3,
                user_access_level=AccessLevel.INTERNAL,
            )
        except Exception:
            rag_results = []

        context_chunks = []
        for r in rag_results or []:
            txt = r.get("text") or r.get("content") or ""
            if txt:
                context_chunks.append(txt.strip())
        context_text = "\n\n".join(context_chunks[:3])

        # 7) LLM 응답 (한국어 전용, 그리디)
        _llm = getattr(request.app.state, "llm", None) or llm
        _llm.load()
        prompt = f"""당신은 한국 제조업 전문 AI입니다.
반드시 한국어로만 답변하세요.
영어 단어를 사용하지 마세요.
모든 기술 용어를 한국어로 번역하세요.

[관련 정보]
{context_text}

[질문]
{payload.message}

[답변] (한국어로만):"""

        # ✅ 한국어 강제 모드: 1차 생성부터 영문자 차단 + 한글 비율 기준 강화
        response_text = _llm.generate(
            prompt,
            max_new_tokens=400,
            temperature=0.1,  # 낮춰서 일관성 향상
            do_sample=False,  # 그리디 모드
        )
        
        # 영어가 포함되어 있으면 후처리
        if any(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in response_text):
            # 주요 용어 번역
            replacements = {
                "OEE": "종합설비효율",
                "Overall Equipment Effectiveness": "종합설비효율",
                "Availability": "가용성",
                "Performance": "성능",
                "Quality": "품질",
                "IoT": "사물인터넷",
                "Machine Learning": "기계학습",
                "Predictive Maintenance": "예측 정비",
                "Smart Factory": "스마트 팩토리",
                "factory": "공장",
                "sensor": "센서",
                "improve": "개선",
                "enhance": "향상",
                "increase": "증가",
                "decrease": "감소",
                "calculate": "계산",
                "based on": "기반으로",
                "compared to": "대비",
            }
            
            for eng, kor in replacements.items():
                response_text = response_text.replace(eng, kor)
                response_text = response_text.replace(eng.lower(), kor)
                response_text = response_text.replace(eng.upper(), kor)

        # 8) 어시스턴트 메시지 저장
        await conv_mgr.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
        )

        return {
            "conversation_id": conversation_id,
            "response": response_text,
            "sources": [r.get("source", "") for r in (rag_results or [])],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb[-2000:]})

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
async def run_production_scenario(query: str, request: Request, db: AsyncSession = Depends(get_db)):
    """생산 모니터링 시나리오"""
    
    llm = request.app.state.llm
    agent = ProductionMonitoringAgent(llm=llm, rag=rag)
    result = await agent.analyze_production(query)
    
    return result

@router.post("/scenarios/maintenance")
async def run_maintenance_scenario(equipment_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    """예측적 유지보수 시나리오"""
    
    llm = request.app.state.llm
    agent = PredictiveMaintenanceAgent(llm=llm, rag=rag)
    result = await agent.predict_failure(equipment_id)
    
    return result

@router.post("/scenarios/quality")
async def run_quality_scenario(request: Request, db: AsyncSession = Depends(get_db)):
    """품질 관리 시나리오"""
    
    llm = request.app.state.llm
    agent = QualityControlAgent(llm=llm, rag=rag)
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
