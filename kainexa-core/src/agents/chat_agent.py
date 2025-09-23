# ============================================
# src/agents/chat_agent.py - 채팅 에이전트
# ============================================
"""src/agents/chat_agent.py"""
from typing import Dict, Any, AsyncGenerator
import asyncio
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.orchestration.graph_executor import GraphExecutor
from src.orchestration.dsl_parser import DSLParser
from src.monitoring.metrics import MetricsManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ChatAgent(BaseAgent):
    """통합 채팅 에이전트"""
    
    def __init__(self):
        super().__init__("chat_agent")
        self.graph_executor = GraphExecutor()
        self.dsl_parser = DSLParser()
        self.metrics = MetricsManager()
        
        # 기본 워크플로우 로드
        self.workflow = self._load_default_workflow()
    
    def _load_default_workflow(self):
        """기본 채팅 워크플로우 로드"""
        workflow_yaml = """
        name: chat_workflow
        steps:
          - step: classify_intent
            type: intent_classify
          - step: retrieve_knowledge
            type: retrieve_knowledge
            condition: "confidence > 0.7"
          - step: generate_response
            type: llm_generate
          - step: execute_actions
            type: mcp_execution
            condition: "requires_action == true"
        """
        return self.dsl_parser.parse_yaml(workflow_yaml)
    
    async def process(self, message: str, user_id: str, 
                     session_id: str, context: Dict = None) -> Dict[str, Any]:
        """메시지 처리"""
        start_time = datetime.now()
        
        try:
            # 실행 컨텍스트 생성
            exec_context = {
                'message': message,
                'user_id': user_id,
                'session_id': session_id,
                'context': context or {},
                'timestamp': start_time
            }
            
            # 워크플로우 실행
            result = await self.graph_executor.execute_graph(
                self.workflow,
                exec_context
            )
            
            # 메트릭 기록
            duration = (datetime.now() - start_time).total_seconds()
            await self.metrics.track_conversation(
                session_id=session_id,
                duration=duration,
                success=True
            )
            
            return {
                'response': result.get('response', '죄송합니다. 처리할 수 없습니다.'),
                'session_id': session_id,
                'intent': result.get('intent'),
                'confidence': result.get('confidence', 0.0),
                'sources': result.get('sources', []),
                'actions': result.get('actions', []),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            await self.metrics.track_error(session_id, str(e))
            raise
    
    async def stream(self, message: str, 
                    session_id: str) -> AsyncGenerator[str, None]:
        """스트리밍 응답"""
        # 청크 단위로 응답 생성
        response = await self.process(message, "", session_id, {})
        
        # 토큰 단위로 스트리밍
        tokens = response['response'].split()
        for token in tokens:
            yield token + " "
            await asyncio.sleep(0.05)  # 스트리밍 효과