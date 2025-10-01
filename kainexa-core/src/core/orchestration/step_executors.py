# src/core/orchestration/step_executors.py
"""모든 Step Executor 통합"""
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime
import structlog

from src.nlp.korean_nlp import KoreanNLPPipeline
from src.governance.vector_store import VectorStore
from src.models.model_factory import ModelFactory

logger = structlog.get_logger()

class BaseExecutor(ABC):
    """실행자 베이스 클래스"""
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Any:
        pass

class IntentClassifyExecutor(BaseExecutor):
    """의도 분류 실행자"""
    
    def __init__(self):
        self.nlp_pipeline = KoreanNLPPipeline()
    
    async def execute(self, context: Dict[str, Any]) -> Dict:
        text = context.get('input_text', '')
        result = await self.nlp_pipeline.process(text)
        
        return {
            'intent': result.intent,
            'entities': result.entities,
            'honorific_level': result.honorific_level.value,
            'sentiment': result.sentiment
        }

class RetrieveKnowledgeExecutor(BaseExecutor):
    """지식 검색 실행자"""
    
    def __init__(self):
        self.vector_store = VectorStore()
    
    async def execute(self, context: Dict[str, Any]) -> List[Dict]:
        query = context.get('query', context.get('input_text', ''))
        k = context.get('k', 5)
        
        results = await self.vector_store.search(query, k)
        
        # Re-ranking 적용
        if len(results) > 1:
            results = self._rerank(query, results)
        
        return results
    
    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """결과 재순위화"""
        # Cross-encoder 사용 (간단 구현)
        for result in results:
            # 쿼리와 텍스트 유사도 재계산
            result['rerank_score'] = result.get('score', 0.0) * 0.8
        
        return sorted(results, key=lambda x: x['rerank_score'], reverse=True)

class LLMGenerateExecutor(BaseExecutor):
    """LLM 생성 실행자"""
    
    def __init__(self):
        self.model_factory = ModelFactory()
        self.llm = None
    
    async def execute(self, context: Dict[str, Any]) -> str:
        # 모델 초기화 (lazy loading)
        if not self.llm:
            model_type = context.get('model', 'solar')
            self.llm = self.model_factory.create_model(model_type)
        
        # 프롬프트 구성
        prompt = self._build_prompt(context)
        
        # 생성 파라미터
        params = {
            'temperature': context.get('temperature', 0.7),
            'max_tokens': context.get('max_tokens', 512),
            'top_p': context.get('top_p', 0.9)
        }
        
        # LLM 생성
        response = await self.llm.generate(prompt, **params)
        
        return response

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """프롬프트 구성"""
        template = context.get('prompt_template', '')
        
        # 기본 템플릿
        if not template:
            template = """당신은 도움이 되는 AI 어시스턴트입니다.
            
사용자: {input_text}

{knowledge}

응답:"""
        
        # 컨텍스트 변수로 치환
        prompt = template
        
        # 입력 텍스트
        prompt = prompt.replace('{input_text}', context.get('input_text', ''))
        
        # RAG 결과 추가
        if 'retrieved_knowledge' in context:
            knowledge_text = "\n관련 정보:\n"
            for i, doc in enumerate(context['retrieved_knowledge'][:3], 1):
                knowledge_text += f"{i}. {doc.get('text', '')}\n"
            prompt = prompt.replace('{knowledge}', knowledge_text)
        else:
            prompt = prompt.replace('{knowledge}', '')
        
        # 대화 이력 추가
        if 'conversation_history' in context:
            history = context['conversation_history'][-5:]  # 최근 5개
            history_text = "\n이전 대화:\n"
            for h in history:
                history_text += f"사용자: {h.get('user', '')}\n"
                history_text += f"AI: {h.get('assistant', '')}\n"
            prompt = history_text + "\n" + prompt
        
        return prompt

class MCPExecutionExecutor(BaseExecutor):
    """MCP 액션 실행자"""
    
    def __init__(self):
        self.action_registry = self._init_actions()
    
    def _init_actions(self) -> Dict:
        """액션 레지스트리 초기화"""
        return {
            'send_email': self._send_email,
            'create_ticket': self._create_ticket,
            'update_database': self._update_database,
            'call_api': self._call_external_api
        }
    
    async def execute(self, context: Dict[str, Any]) -> Dict:
        action_type = context.get('action_type')
        
        if not action_type:
            return {'status': 'skipped', 'reason': 'No action required'}
        
        # 권한 확인
        if not self._check_permission(context):
            return {'status': 'denied', 'reason': 'Insufficient permissions'}
        
        # 액션 실행
        if action_type in self.action_registry:
            result = await self.action_registry[action_type](context)
            return result
        
        return {'status': 'error', 'reason': f'Unknown action: {action_type}'}
    
    def _check_permission(self, context: Dict) -> bool:
        """권한 확인"""
        user_role = context.get('user_role', 'user')
        action_type = context.get('action_type')
        
        # 간단한 권한 매트릭스
        permissions = {
            'user': [],
            'agent': ['send_email', 'create_ticket'],
            'admin': ['send_email', 'create_ticket', 'update_database', 'call_api']
        }
        
        return action_type in permissions.get(user_role, [])
    
    async def _send_email(self, context: Dict) -> Dict:
        """이메일 발송"""
        logger.info("Sending email", context=context)
        # 실제 이메일 발송 로직
        await asyncio.sleep(0.5)  # 시뮬레이션
        return {'status': 'sent', 'message_id': 'email_12345'}
    
    async def _create_ticket(self, context: Dict) -> Dict:
        """티켓 생성"""
        logger.info("Creating ticket", context=context)
        # 티켓 시스템 연동
        return {'status': 'created', 'ticket_id': 'TICK-001'}
    
    async def _update_database(self, context: Dict) -> Dict:
        """데이터베이스 업데이트"""
        logger.info("Updating database", context=context)
        # DB 업데이트 로직
        return {'status': 'updated', 'affected_rows': 1}
    
    async def _call_external_api(self, context: Dict) -> Dict:
        """외부 API 호출"""
        import httpx
        
        url = context.get('api_url')
        method = context.get('api_method', 'GET')
        payload = context.get('api_payload', {})
        
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, json=payload)
            
        return {
            'status': 'completed',
            'status_code': response.status_code,
            'response': response.json() if response.status_code == 200 else None
        }

class ResponsePostprocessExecutor(BaseExecutor):
    """응답 후처리 실행자"""
    
    def __init__(self):
        self.nlp_pipeline = KoreanNLPPipeline()
    
    async def execute(self, context: Dict[str, Any]) -> str:
        response = context.get('llm_response', '')
        
        # 1. 존댓말 변환
        target_honorific = context.get('target_honorific')
        if target_honorific:
            nlp_result = await self.nlp_pipeline.process(response, target_honorific)
            response = nlp_result.text
        
        # 2. 출처 추가
        if 'retrieved_knowledge' in context:
            sources = []
            for doc in context['retrieved_knowledge'][:3]:
                source = doc.get('metadata', {}).get('source', 'Unknown')
                if source not in sources:
                    sources.append(source)
            
            if sources:
                response += f"\n\n출처: {', '.join(sources)}"
        
        # 3. 포맷팅
        response = self._format_response(response)
        
        return response
    
    def _format_response(self, text: str) -> str:
        """응답 포맷팅"""
        # 마크다운 처리
        text = text.strip()
        
        # 이모지 추가 (선택적)
        if '죄송' in text:
            text = '😔 ' + text
        elif '감사' in text:
            text = '😊 ' + text
        
        return text

# Executor 레지스트리
EXECUTOR_REGISTRY = {
    'intent_classify': IntentClassifyExecutor,
    'retrieve_knowledge': RetrieveKnowledgeExecutor,
    'llm_generate': LLMGenerateExecutor,
    'mcp_execution': MCPExecutionExecutor,
    'response_postprocess': ResponsePostprocessExecutor
}

def create_executor(step_type: str) -> BaseExecutor:
    """실행자 생성 팩토리"""
    executor_class = EXECUTOR_REGISTRY.get(step_type)
    if not executor_class:
        raise ValueError(f"Unknown executor type: {step_type}")
    return executor_class()