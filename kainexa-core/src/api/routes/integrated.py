"""
통합 API 엔드포인트 - 한국어 전용
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from time import perf_counter
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import traceback
import logging
from src.core.database import get_db
from src.core.session_manager import SessionManager, ConversationManager
from src.core.models.solar_llm import SolarLLM
from src.core.governance.rag_pipeline import RAGPipeline,DocumentMetadata, AccessLevel, get_rag_pipeline
import os
from src.scenarios.production_monitoring import ProductionMonitoringAgent
from src.scenarios.predictive_maintenance import PredictiveMaintenanceAgent
from src.scenarios.quality_control import QualityControlAgent
from src.api.schemas.schemas import ChatRequest, RagSearchRequest
from uuid import UUID, uuid4
from sqlalchemy import select
from src.core.models.orm_models import Conversation  # 실제 위치에 맞게 임포트

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["integrated"])

# Pydantic 모델
class LoginRequest(BaseModel):
    email: str
    password: str

class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    chunks: int
    status: str

# 전역 인스턴스
llm = SolarLLM()
RAG_CFG = {
    "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
    "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
    "collection_name": os.getenv("QDRANT_COLLECTION", "kainexa_default"),
}
def get_rag_dep() -> RAGPipeline:
    # 필요 시 새 인스턴스 반환(간단), pool/reuse가 필요하면 app.state 등에 캐시
    return get_rag_pipeline(RAG_CFG)

def _parse_uuid_maybe(value: str | None) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(value)
    except Exception:
        return None

async def _ensure_conversation(db: AsyncSession, incoming_id: str | None) -> UUID:
    """
    - 유효한 UUID가 오면: 존재 여부 확인 후 없으면 생성
    - UUID가 아니면: 새 UUID 발급 후 생성
    항상 존재하는 Conversation의 UUID를 반환
    """
    cid = _parse_uuid_maybe(incoming_id)

    # 1) incoming_id가 유효한 UUID인 경우: 존재하면 그대로, 없으면 새로 생성
    if cid:
        row = await db.execute(select(Conversation).where(Conversation.id == cid))
        conv = row.scalar_one_or_none()
        if conv:
            return cid
        conv = Conversation(
            id=cid,
            title=f"대화 {datetime.now():%Y-%m-%d %H:%M}",
            context={},
            status="active",
        )
        db.add(conv)
        await db.flush()
        return cid

    # 2) incoming_id가 없거나(UUID 아님): 새 UUID 발급 후 새 대화 생성
    new_id = uuid4()
    conv = Conversation(
        id=new_id,
        title=f"대화 {datetime.now():%Y-%m-%d %H:%M}",
        context={},
        status="active",
    )
    db.add(conv)
    await db.flush()
    return new_id

@router.post("/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """로그인"""
    from sqlalchemy import select
    from src.core.models import User
    
    query = select(User).where(User.email == request.email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
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

@router.post("/rag/search")
async def rag_search(req: RagSearchRequest, rag: RAGPipeline = Depends(get_rag_dep)):
    results = await rag.search(req.query, k=req.top_k)
    return {"results": results}

@router.post("/chat")
async def chat(
    request_data: ChatRequest,
    request: Request,
    db: AsyncSession = Depends(get_db), 
    rag: RAGPipeline = Depends(get_rag_dep),
):
        
    """채팅 엔드포인트 - 한국어 전용"""
    try:
        
        t0 = perf_counter()
        
        # 사용자 식별
        user_email = (
            request.headers.get("X-User-Email") or 
            request_data.user_email or 
            "demo@kainexa.local"
        )
            
        # 대화 관리자
        conv_manager = ConversationManager(db)

        # 대화 ID 정규화(항상 UUID가 되도록 보장하고, 없으면 생성)
        conversation_uuid = await _ensure_conversation(
            db=db,
            incoming_id=request_data.conversation_id,
        )
        conversation_id = str(conversation_uuid)  # 응답/로그 용 문자열
        
                
        # 사용자 메시지 저장
        await conv_manager.add_message(
            conversation_id=conversation_uuid,  # UUID 객체 O
            role="user",
            content=request_data.message
        )
                   
            
        # RAG 검색 (옵션) ─ 결과를 sources로 풍부화
        rag_context = ""
        rag_results = []
           
        try:
            # 실제 구현 시그니처: (query, access_level)
            rag_results = await rag.search_with_access_control(
            request_data.message, AccessLevel.INTERNAL
            )
        except AttributeError:
            # search_with_access_control 이 없으면 일반 search로 폴백
            rag_results = await rag.search(request_data.message, k=3)
        except Exception as e:
            logger.warning("RAG search failed", exc_info=e)
            rag_results = []

        def _get(d, key, default=None):
            if isinstance(d, dict):
                return d.get(key, default)
            return getattr(d, key, default)

        def _meta(d):
            m = _get(d, "metadata", {}) or {}
            if not isinstance(m, dict):
                m = getattr(m, "dict", lambda: {})()
            return m

        # rag_context 구성도 안전 접근
        if rag_results:
            def _text(r):
                return _get(r, "text", "") or ""
            rag_context = "\n".join([_text(r)[:200] for r in rag_results[:2]])

        sources = [
            {
                "id": _get(r, "id"),
                "score": _get(r, "score"),
                "title": _meta(r).get("title"),
                "source": _meta(r).get("source"),
            }
            for r in (rag_results or [])
        ]     
        
        # LLM 응답 생성
        try:
            # 앱 상태에서 LLM 가져오기 또는 전역 사용
            _llm = getattr(request.app.state, "llm", None) or llm
            _llm.load()
            
            # 한국어 전용 프롬프트
            if rag_context:
                prompt = f"""당신은 한국 제조업 전문 AI 어시스턴트입니다.
반드시 한국어로만 답변하세요. 영어를 사용하지 마세요.

[참고 정보]
{rag_context}

[질문]
{request_data.message}

[답변] 한국어로만:"""
            else:
                prompt = f"""당신은 한국 제조업 전문 AI 어시스턴트입니다.
반드시 한국어로만 답변하세요. 영어를 사용하지 마세요.

[질문]
{request_data.message}

[답변] 한국어로만:"""
            
            response_text = _llm.generate(
                prompt,
                #max_new_tokens=400,
                max_new_tokens = 256 if len(request_data.message) < 100 else 512,
                temperature=0.3,
                do_sample=False  # 그리디
            )
            
            # 응답이 비어있으면 기본 응답                
            if not response_text or not response_text.strip():
                response_text = _get_fallback_response(request_data.message)

            # ① LLM 응답 후처리(이름 치환)
            response_text = _postprocess(response_text, user_email)
            
        except Exception as e:
            # ④ 예외별 폴백 문구 개선
            msg = str(e)
            if "qdrant" in msg.lower():
                logger.error("LLM generation failed (downstream RAG issue)", exc_info=e)
                response_text = "지식검색 서비스에 일시적 문제가 있어요. 대화 자체는 계속 가능합니다."
            else:
                logger.error("LLM generation failed", exc_info=e)
                response_text = "생성 엔진이 잠시 응답이 없습니다. 간단 요약으로 답변드릴게요."
        
        # 어시스턴트 메시지 저장
        await conv_manager.add_message(
            conversation_id=conversation_uuid,
            role="assistant",
            content=response_text
        )
        
        response_text = _postprocess(response_text, user_email)
               
        duration_ms = int((perf_counter() - t0) * 1000)
        return {
            "conversation_id": conversation_id,  # 문자열로 응답
            "response": response_text,
            "sources": sources,                  # ② 풍부화된 소스
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "meta": {
                # model_name 없으면 model_id → pipeline 내부 모델 이름 순으로 폴백
                "model": (
                    getattr(llm, "model_name", None)
                    or getattr(llm, "model_id", None)
                    or getattr(getattr(llm, "pipeline", None), "model_name", None)
                ),
                "latency_ms": duration_ms,
                "rag_used": bool(rag_results),
            },
                      
        }        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        # 에러시에도 기본 응답 반환
        return {
            "conversation_id": "",
            "response": "죄송합니다. 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "sources": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
def _postprocess(text: str, user_email: str | None = None) -> str:
    name = (user_email or "").split("@")[0] or "고객님"
    return (text or "").replace("#@이름#", name)

def _get_fallback_response(query: str) -> str:
    """LLM 실패시 폴백 응답"""
    query_lower = query.lower()
    
    if "oee" in query_lower or "종합설비효율" in query_lower:
        return """스마트 팩토리에서 종합설비효율(OEE)을 개선하는 방법은 다음과 같습니다:

1. **가용성 향상**: 계획된 생산 시간 대비 실제 가동 시간을 늘립니다.
   - 예방 정비를 통한 고장 감소
   - 신속한 고장 대응 체계 구축
   
2. **성능 개선**: 이론적 생산량 대비 실제 생산량을 높입니다.
   - 병목 공정 개선
   - 작업자 숙련도 향상
   
3. **품질 제고**: 전체 생산량 대비 양품 비율을 증가시킵니다.
   - 실시간 품질 모니터링
   - 불량 원인 분석 및 개선

사물인터넷 센서와 기계학습을 활용하면 더욱 효과적인 개선이 가능합니다."""
    
    elif "예측" in query_lower or "정비" in query_lower:
        return """예측적 유지보수는 설비 고장을 사전에 예방하는 스마트한 정비 방법입니다.

주요 특징:
- 센서 데이터 실시간 모니터링
- 고장 패턴 분석 및 예측
- 최적 정비 시점 결정
- 불필요한 정비 비용 절감

이를 통해 설비 가동률을 높이고 유지보수 비용을 절감할 수 있습니다."""
    
    elif "품질" in query_lower or "불량" in query_lower:
        return """품질 관리 개선 방안:

1. **실시간 품질 검사**: 비전 검사 시스템 도입
2. **통계적 공정 관리**: 관리도를 활용한 공정 모니터링
3. **근본 원인 분석**: 불량 발생 원인을 체계적으로 분석
4. **지속적 개선**: PDCA 사이클 적용

이러한 방법들을 통해 불량률을 감소시키고 품질을 향상시킬 수 있습니다."""
    
    else:
        return """제조업 스마트 팩토리와 관련하여 도움을 드릴 수 있습니다.

다음과 같은 주제에 대해 질문해주세요:
- 종합설비효율(OEE) 개선
- 예측적 유지보수
- 품질 관리
- 생산 모니터링
- 스마트 팩토리 구축

구체적인 질문을 주시면 더 자세한 답변을 드리겠습니다."""

# 나머지 엔드포인트들은 동일하게 유지...

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    rag: RAGPipeline = Depends(get_rag_dep),
):
    """문서 업로드"""
    content = await file.read()
    text_content = content.decode('utf-8')
    
    # 1) 반드시 DocumentMetadata 인스턴스로 생성 (필드 자동 필터)
    _raw_meta = {
        "doc_id": f"upload_{datetime.now():%Y%m%d_%H%M%S}",
        "title": file.filename,
        "source": f"upload/{file.filename}",
        "access_level": getattr(AccessLevel.INTERNAL, "value", "INTERNAL"),
        "language": "ko",
    }
    field_names = set(
        getattr(DocumentMetadata, "model_fields", {}).keys()
        or getattr(DocumentMetadata, "__fields__", {}).keys()
        or {"doc_id", "title", "source", "access_level", "language"}
    )
    _filtered = {k: v for k, v in _raw_meta.items() if k in field_names}
    metadata = DocumentMetadata(**_filtered)

    # 2) RAGPipeline 표준 경로: add_document(content, metadata)
    #    ※ metadata는 반드시 DocumentMetadata 인스턴스여야 함
    success = await rag.add_document(text_content, metadata)
    if not success:
        raise HTTPException(status_code=500, detail="Ingest failed: add_document returned False")

    # 응답 구성 (안전 추출)
    doc_id = meta_dict.get("doc_id") or getattr(metadata, "doc_id", None) or ""
    title = meta_dict.get("title") or getattr(metadata, "title", None) or file.filename
    chunks = max(1, len(text_content) // 500)

    return DocumentUploadResponse(
        document_id=doc_id,
        title=title,
        chunks=chunks,
        status="success",
    )

@router.get("/documents/search")
async def search_documents(
    query: str,
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
    rag: RAGPipeline = Depends(get_rag_dep),
):
    """문서 검색"""  
    results = await rag.search_with_access_control(query, AccessLevel.INTERNAL)
    
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }
    
@router.post("/documents/search")
async def search_documents_post(
    body: dict,
    db: AsyncSession = Depends(get_db),
    rag: RAGPipeline = Depends(get_rag_dep),
):
    """
    문서 검색 (POST, JSON 본문)
    기대 형식: {"query": "...", "top_k": 3}
    """
    query = (body or {}).get("query") or ""
    top_k = int((body or {}).get("top_k") or 5)
    if not query:
        raise HTTPException(status_code=422, detail="`query` is required")
    # 현재 RAG 구현이 (query, access_level) 시그니처 → top_k는 내부 기본값 사용
    #try:
    #    results = await rag.search_with_access_control(query, AccessLevel.INTERNAL)
    #except AttributeError:
    #    # 메서드가 없으면 일반 search로 폴백
    #    try:
    #        results = await rag.search(query, k=top_k)
    #    except TypeError:
    #        results = await rag.search(query, top_k=top_k)
    
    # ❶ 인덱싱/검색 파이프라인이 정상인지 raw search로 먼저 확인
    try:
        results = await rag.search(query, k=top_k)   # ← ACL 미적용
    except TypeError:
        results = await rag.search(query, top_k=top_k)
    
    return {"query": query, "results": results, "count": len(results)}


@router.post("/scenarios/production")
async def run_production_scenario(
    query: str = "생산 현황 보고해줘",
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    rag: RAGPipeline = Depends(get_rag_dep),
):
    """생산 모니터링 시나리오"""
    try:
        _llm = getattr(request.app.state, "llm", None) if request else None
        agent = ProductionMonitoringAgent(llm=_llm or llm, rag=rag)
        result = await agent.analyze_production(query)
        return result
    except Exception as e:
        logger.error(f"Production scenario error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenarios/maintenance")
async def run_maintenance_scenario(
    equipment_id: str = "CNC_007",
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    rag: RAGPipeline = Depends(get_rag_dep),
):
    """예측적 유지보수 시나리오"""
    try:
        _llm = getattr(request.app.state, "llm", None) if request else None
        agent = PredictiveMaintenanceAgent(llm=_llm or llm, rag=rag)
        result = await agent.predict_failure(equipment_id)
        return result
    except Exception as e:
        logger.error(f"Maintenance scenario error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenarios/quality")
async def run_quality_scenario(
    request: Request = None,
    db: AsyncSession = Depends(get_db),
    rag: RAGPipeline = Depends(get_rag_dep),
):
    """품질 관리 시나리오"""
    try:
        _llm = getattr(request.app.state, "llm", None) if request else None
        agent = QualityControlAgent(llm=_llm or llm, rag=rag)
        result = await agent.analyze_quality()
        return result
    except Exception as e:
        logger.error(f"Quality scenario error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """대화 기록 조회"""
    conv_manager = ConversationManager(db)
        
    cid = _parse_uuid_maybe(conversation_id)
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id must be a valid UUID")
    messages = await conv_manager.get_conversation_history(str(cid), limit=limit)
    
    
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
async def full_health_check(rag: RAGPipeline = Depends(get_rag_dep)):
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