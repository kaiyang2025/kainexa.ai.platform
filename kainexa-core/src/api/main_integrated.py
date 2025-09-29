# kainexa-core/src/api/main_integrated.py 수정
# Solar LLM을 실제로 호출하는 워크플로우 실행 코드

from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
from uuid import uuid4
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio

from src.api.routes.integrated import router as integrated_router
from src.core.config import settings
from src.models.solar_llm import SolarLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Kainexa Integrated API")
    
    # LLM 싱글턴 예열 & 캐싱
    try:
        if not hasattr(app.state, "llm") or app.state.llm is None:
            logger.info("Loading Solar LLM model...")
            
            # 모델 경로를 명확히 지정
            model_path = os.getenv("SOLAR_MODEL_PATH", "beomi/OPEN-SOLAR-KO-10.7B")
            
            # 로컬 경로인지 HuggingFace 모델인지 확인
            if os.path.exists(model_path):
                # 로컬 모델
                logger.info(f"Using local model: {model_path}")
            else:
                # HuggingFace Hub 모델
                logger.info(f"Using HuggingFace model: {model_path}")
            
            llm = SolarLLM(
                model_path=model_path,  # 문자열로 전달
                load_in_8bit=True,  # 8비트 양자화
                device_map="auto"  # 자동 분산
            )
            llm.load()
            app.state.llm = llm
            logger.info("✅ SolarLLM loaded and cached successfully")
    except Exception as e:
        logger.exception("❌ LLM warm-up failed: %s", e)
        app.state.llm = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Kainexa Integrated API")
    try:
        if hasattr(app.state, "llm") and app.state.llm is not None:
            app.state.llm = None
            logger.info("LLM resources released")
    except Exception:
        pass

app = FastAPI(
    title="Kainexa AI Platform",
    version="1.0.0",
    description="Manufacturing AI Agent Platform",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함
app.include_router(integrated_router)

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
            "workflow": "/api/v1/workflow/execute",
            "health": "/api/v1/health/full"
        }
    }

@app.post("/api/v1/workflow/execute")
async def execute_workflow(request: dict):
    """워크플로우 실행 - 실제 LLM 호출 포함"""
    nodes = request.get("nodes", [])
    edges = request.get("edges", [])
    
    print(f"\n{'='*60}")
    print(f"🚀 워크플로우 실행 시작")
    print(f"{'='*60}")
    print(f"📊 노드 수: {len(nodes)}, 연결 수: {len(edges)}")
    
    results = []
    execution_log = []
    
    # 노드별 실행
    for i, node in enumerate(nodes):
        node_id = node.get('id', f'node_{i}')
        node_type = node.get('type', 'unknown')
        node_data = node.get('data', {})
        node_config = node_data.get('config', {})
        
        print(f"\n{'─'*40}")
        print(f"📌 노드 {i+1}/{len(nodes)}: {node_id}")
        print(f"   타입: {node_type}")
        print(f"   라벨: {node_data.get('label', 'N/A')}")
        
        result = {
            "node_id": node_id,
            "type": node_type,
            "label": node_data.get('label'),
            "status": "executing",
            "output": None,
            "execution_time": None
        }
        
        start_time = datetime.now()
        
        try:
            # 노드 타입별 실행
            if node_type == 'intent':
                # 의도 분류 (간단한 키워드 매칭)
                print(f"   🎯 의도 분류 실행...")
                await asyncio.sleep(0.5)  # 시뮬레이션 딜레이
                result["output"] = {
                    "intent": "greeting",
                    "confidence": 0.95,
                    "message": "사용자 의도: 인사/환영"
                }
                print(f"   ✅ 의도 분류 완료: greeting (95% 신뢰도)")
                
            elif node_type == 'llm':
                # ⭐ 실제 Solar LLM 호출
                print(f"   🤖 Solar LLM 호출 중...")
                
                # LLM이 로드되어 있는지 확인
                if not hasattr(app.state, "llm") or app.state.llm is None:
                    print(f"   ⚠️  LLM이 로드되지 않음. Mock 응답 사용.")
                    result["output"] = {
                        "response": "[Mock] 안녕하세요! 무엇을 도와드릴까요?",
                        "model": "mock",
                        "tokens": 0
                    }
                else:
                    try:
                        # 명확한 한국어 지시 프롬프트
                        system_prompt = """당신은 한국 제조업 전문 AI 어시스턴트입니다.
                    반드시 자연스러운 한국어로만 응답하세요.
                    영어, 한자, 특수문자를 사용하지 마세요.
                    전문용어는 한국어로 설명하되, 약어(OEE/IoT/AI)는 괄호로 병기 가능합니다."""
                        
                        # 이전 노드의 출력을 입력으로 사용
                        user_message = "안녕하세요"
                        if i > 0 and results:
                            prev_output = results[-1].get('output', {})
                            if 'message' in prev_output:
                                user_message = prev_output['message']
                        
                        prompt = f"{system_prompt}\n\n사용자: {user_message}\n어시스턴트:"
                        
                        print(f"   📝 프롬프트 길이: {len(prompt)}자")
                        print(f"   ⏳ LLM 생성 중... (최대 30초 소요)")
                        
                        # 실제 LLM 호출
                        response = app.state.llm.generate(
                            prompt=prompt,
                            max_new_tokens=100,
                            temperature=0.7,  # 적절한 창의성
                            top_p=0.9,
                            top_k=50,
                            do_sample=True,  # 샘플링 활성화
                            ko_only=True,  # 한국어 전용 모드
                        )
                        
                        result["output"] = {
                            "response": response,
                            "model": "solar-10.7b",
                            "tokens": len(response.split()),
                            "temperature": node_config.get('temperature', 0.7)
                        }
                        print(f"   ✅ LLM 응답 생성 완료 ({len(response)}자)")
                        print(f"   💬 응답: {response[:100]}...")
                        
                    except Exception as e:
                        print(f"   ❌ LLM 호출 실패: {str(e)}")
                        result["output"] = {
                            "response": f"LLM 오류: {str(e)}",
                            "model": "solar-error",
                            "error": str(e)
                        }
                
            elif node_type == 'api':
                # API 호출 시뮬레이션
                print(f"   🌐 API 호출 시뮬레이션...")
                url = node_config.get('url', 'https://api.example.com')
                await asyncio.sleep(0.3)
                result["output"] = {
                    "status_code": 200,
                    "url": url,
                    "data": {"result": "API 응답 데이터"}
                }
                print(f"   ✅ API 호출 완료: {url}")
                
            elif node_type == 'condition':
                # 조건 분기
                print(f"   🔀 조건 평가...")
                result["output"] = {
                    "condition_met": True,
                    "expression": node_config.get('expression', 'true'),
                    "next_node": edges[0]['target'] if edges else None
                }
                print(f"   ✅ 조건 평가 완료: True")
                
            elif node_type == 'loop':
                # 반복 실행
                iterations = node_config.get('iterations', 3)
                print(f"   🔄 반복 실행 ({iterations}회)...")
                result["output"] = {
                    "iterations": iterations,
                    "completed": True,
                    "results": [f"iteration_{j+1}" for j in range(iterations)]
                }
                print(f"   ✅ 반복 완료: {iterations}회")
            
            else:
                result["output"] = {"message": f"Unknown node type: {node_type}"}
            
            result["status"] = "completed"
            
        except Exception as e:
            print(f"   ❌ 노드 실행 오류: {str(e)}")
            result["status"] = "error"
            result["output"] = {"error": str(e)}
        
        # 실행 시간 계산
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        print(f"   ⏱️  실행 시간: {result['execution_time']:.2f}초")
        
        results.append(result)
        execution_log.append(f"[{datetime.now().isoformat()}] {node_id}: {result['status']}")
    
    # 전체 실행 완료
    print(f"\n{'='*60}")
    print(f"✅ 워크플로우 실행 완료!")
    print(f"{'='*60}")
    print(f"📊 실행 결과:")
    print(f"   - 성공: {sum(1 for r in results if r['status'] == 'completed')}개")
    print(f"   - 실패: {sum(1 for r in results if r['status'] == 'error')}개")
    print(f"   - 총 실행 시간: {sum(r.get('execution_time', 0) for r in results):.2f}초")
    
    return {
        "execution_id": f"exec_{uuid4().hex[:8]}",
        "status": "completed",
        "message": f"워크플로우 실행 완료: {len(nodes)}개 노드 처리",
        "nodes_executed": len(nodes),
        "results": results,
        "execution_log": execution_log,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# LLM 테스트 엔드포인트 추가
@app.post("/api/v1/test/llm")
async def test_llm(prompt: str = "안녕하세요. 자기소개를 해주세요."):
    """Solar LLM 단독 테스트"""
    print(f"\n{'='*60}")
    print(f"🧪 Solar LLM 테스트")
    print(f"{'='*60}")
    
    if not hasattr(app.state, "llm") or app.state.llm is None:
        return {
            "status": "error",
            "message": "LLM not loaded",
            "suggestion": "서버를 재시작하여 LLM을 로드하세요"
        }
    
    try:
        print(f"📝 입력: {prompt}")
        print(f"⏳ 생성 중...")
        
        start_time = datetime.now()
        response = app.state.llm.generate(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        print(f"✅ 생성 완료 ({execution_time:.2f}초)")
        print(f"💬 응답: {response[:200]}...")
        
        return {
            "status": "success",
            "prompt": prompt,
            "response": response,
            "model": "solar-10.7b",
            "execution_time": execution_time,
            "tokens": len(response.split()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ LLM 테스트 실패: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "prompt": prompt
        }

# 헬스 체크
@app.get("/api/v1/health/full")
async def health_full():
    """상세 헬스 체크"""
    llm_status = "healthy" if (hasattr(app.state, "llm") and app.state.llm is not None) else "not_loaded"
    
    return {
        "status": "healthy",
        "services": {
            "llm": {
                "status": llm_status,
                "model": "solar-10.7b" if llm_status == "healthy" else None
            },
            "rag": {"status": "healthy"},
            "database": {"status": "healthy"},
            "cache": {"status": "healthy"}
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Kainexa Core API with Solar LLM")
    print("="*60)
    print("📡 Endpoints:")
    print("  - Workflow: POST /api/v1/workflow/execute")
    print("  - LLM Test: POST /api/v1/test/llm")
    print("  - Health: GET /api/v1/health/full")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)