# src/scenarios/employee_workflow.py
"""
직원 A의 업무 연속성 시나리오 구현
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
from dataclasses import dataclass

from src.orchestration.dsl_parser import DSLParser
from src.orchestration.graph_executor import GraphExecutor, ExecutionContext
from src.auth.mcp_permissions import MCPAuthManager, Role, Resource, Permission
from src.governance.rag_pipeline import RAGGovernance, AccessLevel
from src.models.tensor_parallel import TensorParallelSolarLLM

@dataclass
class EmployeeSession:
    """직원 세션 관리"""
    employee_id: str
    name: str
    role: str
    department: str
    last_activity: datetime
    context_history: List[Dict[str, Any]]
    current_task: Optional[str] = None
    preferences: Dict[str, Any] = None

class AgenticWorkflowManager:
    """Agentic AI 워크플로우 매니저"""
    
    def __init__(self):
        # 핵심 컴포넌트
        self.auth_manager = MCPAuthManager(secret_key="kainexa-secret")
        self.rag_governance = RAGGovernance()
        self.llm = TensorParallelSolarLLM()
        
        # 세션 관리 (실제로는 Redis 사용)
        self.sessions: Dict[str, EmployeeSession] = {}
        
        # 워크플로우 DSL
        self.workflow_dsl = self._load_workflow_dsl()
        
    def _load_workflow_dsl(self) -> str:
        """워크플로우 DSL 정의"""
        return """
        name: employee_continuation_workflow
        version: "1.0"
        
        policies:
          session_timeout: 86400  # 24시간
          context_window: 10  # 최근 10개 대화 유지
          
        graph:
          # 1. 직원 인증 및 식별
          - step: authenticate_employee
            type: auth_check
            params:
              verify_token: true
              load_profile: true
            policy:
              if: "not authenticated"
              then:
                action: redirect_login
                
          # 2. 이전 컨텍스트 복원
          - step: restore_context
            type: context_retrieval
            params:
              load_history: true
              load_last_task: true
              load_preferences: true
            cache: true
            
          # 3. 작업 의도 파악
          - step: classify_intent
            type: intent_classify
            params:
              model: "solar-10.7b"
              use_context: true
              
          # 4. RAG 기반 정보 검색
          - step: retrieve_information
            type: rag_retrieval
            params:
              sources: ["sales_reports", "internal_docs"]
              time_filter: "last_month"
              access_level: "employee"
            policy:
              if: "intent == 'sales_inquiry'"
              then:
                retrieve_specific: "sales_metrics"
                
          # 5. LLM 응답 생성
          - step: generate_response
            type: llm_generate
            params:
              model: "solar-10.7b"
              temperature: 0.3
              use_rag_context: true
              use_session_context: true
              prompt_template: |
                당신은 {employee_name}님의 업무 어시스턴트입니다.
                
                이전 작업 컨텍스트:
                {previous_context}
                
                현재 질문: {current_query}
                
                관련 정보 (RAG):
                {rag_results}
                
                적절한 응답을 생성하세요.
                
          # 6. MCP 액션 실행
          - step: execute_actions
            type: mcp_execution
            params:
              check_permissions: true
              audit_log: true
            policy:
              if: "requires_action"
              then:
                validate_permission: true
                execute_with_confirmation: true
                
          # 7. 컨텍스트 업데이트
          - step: update_context
            type: context_update
            params:
              save_interaction: true
              update_task_status: true
              update_preferences: true
        """
        
    async def handle_employee_interaction(self, 
                                         employee_id: str,
                                         message: str,
                                         token: str) -> Dict[str, Any]:
        """직원 상호작용 처리"""
        
        # 1. 직원 인증 및 세션 확인
        employee_session = await self._authenticate_and_load_session(
            employee_id, token
        )
        
        # 2. 이전 작업 컨텍스트 복원
        context = await self._restore_context(employee_session)
        
        # 3. 현재 메시지 처리
        response = await self._process_message(
            employee_session,
            message,
            context
        )
        
        # 4. 세션 업데이트
        await self._update_session(employee_session, message, response)
        
        return response
    
    async def _authenticate_and_load_session(self, 
                                            employee_id: str,
                                            token: str) -> EmployeeSession:
        """직원 인증 및 세션 로드"""
        
        # MCP 토큰 검증
        token_payload = self.auth_manager.verify_token(token)
        
        # 기존 세션 확인
        if employee_id in self.sessions:
            session = self.sessions[employee_id]
            
            # 세션 타임아웃 체크
            if (datetime.now() - session.last_activity).seconds < 86400:
                print(f"✅ 직원 {session.name}님, 다시 오셨네요!")
                session.last_activity = datetime.now()
                return session
        
        # 새 세션 생성
        session = EmployeeSession(
            employee_id=employee_id,
            name=token_payload.metadata.get('name', 'Unknown'),
            role=token_payload.role.value,
            department=token_payload.metadata.get('department', 'General'),
            last_activity=datetime.now(),
            context_history=[],
            preferences={}
        )
        
        self.sessions[employee_id] = session
        print(f"👋 새로운 세션 생성: {session.name}님")
        
        return session
    
    async def _restore_context(self, 
                              session: EmployeeSession) -> Dict[str, Any]:
        """이전 작업 컨텍스트 복원"""
        
        context = {
            'employee_name': session.name,
            'department': session.department,
            'last_task': session.current_task,
            'history': []
        }
        
        # 최근 대화 내역 (최대 10개)
        if session.context_history:
            recent_history = session.context_history[-10:]
            context['history'] = recent_history
            
            # 마지막 작업 요약
            last_interaction = recent_history[-1]
            context['last_interaction'] = {
                'timestamp': last_interaction.get('timestamp'),
                'query': last_interaction.get('query'),
                'task': last_interaction.get('task')
            }
            
            print(f"📝 이전 작업 복원: {session.current_task}")
            
        return context
    
    async def _process_message(self,
                              session: EmployeeSession,
                              message: str,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """메시지 처리 및 응답 생성"""
        
        print(f"💬 처리 중: {message}")
        
        # 의도 파악
        intent = await self._classify_intent(message, context)
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'intent': intent,
            'rag_results': None,
            'llm_response': None,
            'actions': []
        }
        
        # RAG 검색 필요 여부 확인
        if self._needs_rag_retrieval(intent, message):
            print("🔍 RAG에서 정보 검색 중...")
            
            # 지난달 매출 정보 등 검색
            rag_results = await self._retrieve_from_rag(
                message, 
                session,
                time_filter="last_month"
            )
            response['rag_results'] = rag_results
            
            # 검색 결과 요약
            if rag_results:
                print(f"📊 {len(rag_results)}개 관련 문서 발견")
        
        # LLM 응답 생성
        llm_response = await self._generate_llm_response(
            message,
            context,
            response.get('rag_results'),
            session
        )
        response['llm_response'] = llm_response
        
        # MCP 액션 실행 필요 확인
        if self._needs_mcp_action(intent, message):
            print("⚡ MCP 액션 실행...")
            
            actions = await self._execute_mcp_actions(
                intent,
                message,
                session
            )
            response['actions'] = actions
        
        return response
    
    async def _classify_intent(self, 
                              message: str,
                              context: Dict[str, Any]) -> str:
        """의도 분류"""
        
        # 간단한 키워드 기반 분류 (실제로는 ML 모델 사용)
        intent_keywords = {
            'sales_inquiry': ['매출', '판매', '실적', '수익'],
            'task_continuation': ['어제', '이전', '계속', '이어서'],
            'report_generation': ['보고서', '리포트', '작성', '문서'],
            'data_analysis': ['분석', '통계', '차트', '그래프'],
            'action_request': ['실행', '처리', '진행', '수행']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in message for keyword in keywords):
                return intent
                
        return 'general_inquiry'
    
    def _needs_rag_retrieval(self, intent: str, message: str) -> bool:
        """RAG 검색 필요 여부"""
        rag_intents = ['sales_inquiry', 'data_analysis', 'report_generation']
        rag_keywords = ['정보', '데이터', '자료', '문서', '매출', '실적']
        
        return (intent in rag_intents or 
                any(keyword in message for keyword in rag_keywords))
    
    async def _retrieve_from_rag(self,
                                message: str,
                                session: EmployeeSession,
                                time_filter: str = None) -> List[Dict[str, Any]]:
        """RAG에서 정보 검색"""
        
        # 접근 권한 레벨 결정
        access_level = AccessLevel.INTERNAL
        if session.role in ['admin', 'manager']:
            access_level = AccessLevel.CONFIDENTIAL
            
        # 필터 설정
        filters = {}
        if time_filter == "last_month":
            last_month = datetime.now() - timedelta(days=30)
            filters['created_after'] = last_month.isoformat()
            
        # RAG 검색
        results = await self.rag_governance.retrieve(
            query=message,
            k=5,
            user_access_level=access_level,
            filters=filters
        )
        
        return results
    
    async def _generate_llm_response(self,
                                    message: str,
                                    context: Dict[str, Any],
                                    rag_results: Optional[List[Dict]],
                                    session: EmployeeSession) -> str:
        """LLM 응답 생성"""
        
        # 프롬프트 구성
        prompt = f"""
        당신은 {session.name}님의 업무 어시스턴트입니다.
        부서: {session.department}
        역할: {session.role}
        
        이전 작업: {session.current_task or '없음'}
        마지막 대화: {context.get('last_interaction', {}).get('query', '없음')}
        
        현재 질문: {message}
        """
        
        if rag_results:
            prompt += "\n\n관련 정보:\n"
            for i, result in enumerate(rag_results[:3], 1):
                prompt += f"{i}. {result.get('text', '')}\n"
                prompt += f"   출처: {result.get('source', '')}\n"
        
        prompt += "\n위 정보를 바탕으로 적절한 응답을 생성하세요."
        
        # LLM 생성
        response = await self.llm.generate(
            prompt=prompt,
            max_length=512,
            temperature=0.3
        )
        
        return response['text']
    
    def _needs_mcp_action(self, intent: str, message: str) -> bool:
        """MCP 액션 필요 여부"""
        action_intents = ['action_request', 'report_generation']
        action_keywords = ['실행', '처리', '생성', '만들어', '보내']
        
        return (intent in action_intents or 
                any(keyword in message for keyword in action_keywords))
    
    async def _execute_mcp_actions(self,
                                  intent: str,
                                  message: str,
                                  session: EmployeeSession) -> List[Dict[str, Any]]:
        """MCP 액션 실행"""
        
        actions = []
        
        # 권한 확인
        can_execute = self.auth_manager.check_permission(
            session,
            Resource.ACTION,
            Permission.EXECUTE
        )
        
        if not can_execute:
            actions.append({
                'type': 'permission_denied',
                'message': '해당 작업을 실행할 권한이 없습니다.'
            })
            return actions
        
        # 액션 타입별 처리
        if '보고서' in message:
            actions.append({
                'type': 'generate_report',
                'status': 'initiated',
                'details': '월간 매출 보고서 생성 중...'
            })
            
        elif '이메일' in message:
            actions.append({
                'type': 'send_email',
                'status': 'prepared',
                'details': '이메일 초안 작성 완료'
            })
            
        elif '분석' in message:
            actions.append({
                'type': 'data_analysis',
                'status': 'processing',
                'details': '데이터 분석 진행 중...'
            })
        
        return actions
    
    async def _update_session(self,
                             session: EmployeeSession,
                             message: str,
                             response: Dict[str, Any]):
        """세션 업데이트"""
        
        # 대화 기록 추가
        session.context_history.append({
            'timestamp': response['timestamp'],
            'query': message,
            'response': response['llm_response'],
            'intent': response['intent'],
            'has_rag': response['rag_results'] is not None,
            'has_actions': len(response['actions']) > 0
        })
        
        # 현재 작업 업데이트
        if response['intent'] == 'task_continuation':
            session.current_task = f"Continuing: {message[:50]}..."
        elif response['intent'] in ['report_generation', 'data_analysis']:
            session.current_task = f"Working on: {response['intent']}"
            
        # 마지막 활동 시간 업데이트
        session.last_activity = datetime.now()
        
        print(f"💾 세션 업데이트 완료")


# 시나리오 실행 예제
async def run_scenario():
    """직원 A 시나리오 실행"""
    
    print("=" * 60)
    print("Kainexa AI Agent Platform - Employee Workflow Scenario")
    print("=" * 60)
    
    # 워크플로우 매니저 초기화
    manager = AgenticWorkflowManager()
    
    # 직원 A 토큰 생성 (첫 로그인)
    token = manager.auth_manager.create_token(
        user_id="employee_a",
        role=Role.AGENT,
        permissions=None
    )
    
    print("\n[Day 1 - 첫 작업]")
    print("-" * 40)
    
    # Day 1: 첫 작업
    response1 = await manager.handle_employee_interaction(
        employee_id="employee_a",
        message="지난달 매출 정보를 분석해줘",
        token=token
    )
    
    print(f"Intent: {response1['intent']}")
    print(f"RAG 검색: {'Yes' if response1['rag_results'] else 'No'}")
    print(f"Response: {response1['llm_response'][:200]}...")
    
    await asyncio.sleep(1)
    
    print("\n[Day 2 - 작업 연속]")
    print("-" * 40)
    
    # Day 2: 어제 작업 이어가기
    response2 = await manager.handle_employee_interaction(
        employee_id="employee_a",
        message="어제 분석하던 매출 정보를 이어서 보고서로 만들어줘",
        token=token
    )
    
    print(f"이전 작업 인식: Yes")
    print(f"Intent: {response2['intent']}")
    print(f"MCP Actions: {response2['actions']}")
    print(f"Response: {response2['llm_response'][:200]}...")
    
    print("\n✅ 시나리오 실행 완료!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_scenario())