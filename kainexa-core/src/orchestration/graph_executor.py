# src/orchestration/graph_executor.py
import asyncio
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import structlog
from abc import ABC, abstractmethod

from .dsl_parser import GraphConfig, StepConfig, StepType
from .policy_engine import PolicyEngine, PolicyDecision
from ..monitoring.metrics_collector import MetricsCollector

logger = structlog.get_logger()

@dataclass
class ExecutionContext:
    """실행 컨텍스트"""
    session_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    
    def set_variable(self, key: str, value: Any):
        """변수 설정"""
        self.variables[key] = value
        
    def get_variable(self, key: str, default: Any = None) -> Any:
        """변수 가져오기"""
        return self.variables.get(key, default)
        
    def add_history(self, step_name: str, result: Any, duration: float):
        """실행 이력 추가"""
        self.history.append({
            'step': step_name,
            'result': result,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_error(self, step_name: str, error: Exception):
        """에러 추가"""
        self.errors.append({
            'step': step_name,
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': datetime.now().isoformat()
        })

class StepExecutor(ABC):
    """단계 실행자 베이스 클래스"""
    
    @abstractmethod
    async def execute(self, step: StepConfig, context: ExecutionContext) -> Any:
        """단계 실행"""
        pass

class IntentClassifyExecutor(StepExecutor):
    """의도 분류 실행자"""
    
    def __init__(self, nlp_pipeline):
        self.nlp_pipeline = nlp_pipeline
        
    async def execute(self, step: StepConfig, context: ExecutionContext) -> Dict[str, Any]:
        """의도 분류 실행"""
        text = context.get_variable('input_text', '')
        
        result = await self.nlp_pipeline.classify_intent(
            text,
            model=step.params.get('model', 'default'),
            threshold=step.params.get('threshold', 0.5)
        )
        
        context.set_variable('intent', result['intent'])
        context.set_variable('confidence', result['confidence'])
        
        return result

class RetrieveKnowledgeExecutor(StepExecutor):
    """지식 검색 실행자"""
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        
    async def execute(self, step: StepConfig, context: ExecutionContext) -> List[Dict]:
        """지식 검색 실행"""
        query = context.get_variable('query', context.get_variable('input_text', ''))
        
        results = await self.knowledge_base.retrieve(
            query,
            k=step.params.get('k', 5),
            threshold=step.params.get('threshold', 0.7)
        )
        
        context.set_variable('retrieved_knowledge', results)
        
        return results

class LLMGenerateExecutor(StepExecutor):
    """LLM 생성 실행자"""
    
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        
    async def execute(self, step: StepConfig, context: ExecutionContext) -> str:
        """LLM 텍스트 생성"""
        prompt = self._build_prompt(step, context)
        
        response = await self.llm_handler.generate(
            prompt,
            model=step.params.get('model', 'solar-10.7b'),
            temperature=step.params.get('temperature', 0.7),
            max_tokens=step.params.get('max_tokens', 512)
        )
        
        context.set_variable('llm_response', response)
        
        return response
        
    def _build_prompt(self, step: StepConfig, context: ExecutionContext) -> str:
        """프롬프트 구성"""
        template = step.params.get('prompt_template', '{input_text}')
        
        # 컨텍스트 변수로 템플릿 채우기
        for key, value in context.variables.items():
            template = template.replace(f'{{{key}}}', str(value))
            
        # 지식베이스 결과 추가
        if 'retrieved_knowledge' in context.variables:
            knowledge = context.variables['retrieved_knowledge']
            knowledge_text = '\n'.join([doc['text'] for doc in knowledge[:3]])
            template = template.replace('{knowledge}', knowledge_text)
            
        return template

class GraphExecutor:
    """그래프 실행 엔진"""
    
    def __init__(self, 
                 policy_engine: PolicyEngine,
                 metrics_collector: MetricsCollector,
                 step_executors: Dict[StepType, StepExecutor]):
        self.policy_engine = policy_engine
        self.metrics = metrics_collector
        self.step_executors = step_executors
        self.execution_cache = {}
        
    async def execute_graph(self, 
                           graph_config: GraphConfig,
                           initial_context: Optional[ExecutionContext] = None) -> ExecutionContext:
        """그래프 실행"""
        
        # 실행 컨텍스트 초기화
        context = initial_context or ExecutionContext(
            session_id=f"session_{time.time()}"
        )
        
        # 진입점부터 실행
        current_step = graph_config.entry_point
        visited_steps: Set[str] = set()
        
        logger.info(f"Starting graph execution: {graph_config.name}",
                   session_id=context.session_id)
        
        try:
            while current_step and current_step != 'end':
                # 순환 참조 방지
                if current_step in visited_steps:
                    logger.warning(f"Circular reference detected at step: {current_step}")
                    break
                    
                visited_steps.add(current_step)
                
                # 현재 단계 실행
                if current_step in graph_config.steps:
                    step = graph_config.steps[current_step]
                    
                    # 정책 확인
                    policy_decision = await self.policy_engine.evaluate(
                        step, context, graph_config.policies
                    )
                    
                    if policy_decision.action == 'skip':
                        logger.info(f"Skipping step {current_step} by policy")
                        current_step = self._get_next_step(step, context)
                        continue
                        
                    if policy_decision.action == 'escalate':
                        logger.info(f"Escalating at step {current_step}")
                        context.set_variable('escalated', True)
                        break
                        
                    # 단계 실행
                    result = await self._execute_step(step, context)
                    
                    # 다음 단계 결정
                    current_step = self._get_next_step(step, context, result)
                else:
                    logger.error(f"Unknown step: {current_step}")
                    break
                    
        except Exception as e:
            logger.error(f"Graph execution failed: {e}", 
                        session_id=context.session_id)
            context.add_error('graph_execution', e)
            
        # 실행 완료 메트릭
        total_duration = time.time() - context.start_time
        await self.metrics.track_conversation(
            session_id=context.session_id,
            message_count=len(context.history),
            intent=context.get_variable('intent', 'unknown'),
            resolved=not context.errors
        )
        
        logger.info(f"Graph execution completed in {total_duration:.2f}s",
                   session_id=context.session_id,
                   steps_executed=len(visited_steps))
        
        return context
        
    async def _execute_step(self, 
                          step: StepConfig, 
                          context: ExecutionContext) -> Any:
        """단일 단계 실행"""
        
        start_time = time.time()
        result = None
        
        try:
            # 캐시 확인
            if step.cache:
                cache_key = self._get_cache_key(step, context)
                if cache_key in self.execution_cache:
                    logger.debug(f"Using cached result for step {step.name}")
                    return self.execution_cache[cache_key]
                    
            # 병렬 실행
            if step.parallel and step.type == StepType.PARALLEL:
                result = await self._execute_parallel(step, context)
            else:
                # 실행자 선택
                executor = self.step_executors.get(step.type)
                if not executor:
                    raise ValueError(f"No executor for step type: {step.type}")
                    
                # 타임아웃과 재시도 적용
                result = await self._execute_with_retry(
                    executor, step, context
                )
                
            # 캐시 저장
            if step.cache and result is not None:
                cache_key = self._get_cache_key(step, context)
                self.execution_cache[cache_key] = result
                
            # 실행 이력 추가
            duration = time.time() - start_time
            context.add_history(step.name, result, duration)
            
            # 메트릭 수집
            if step.type == StepType.LLM_GENERATE:
                await self.metrics.track_llm_inference(
                    model=step.params.get('model', 'default'),
                    tokens=len(str(result).split()),
                    duration=duration
                )
                
            return result
            
        except Exception as e:
            logger.error(f"Step execution failed: {step.name}", error=str(e))
            context.add_error(step.name, e)
            
            # 재시도 또는 폴백
            if step.retry > 0:
                logger.info(f"Retrying step {step.name}")
                step.retry -= 1
                return await self._execute_step(step, context)
                
            raise
            
    async def _execute_with_retry(self, 
                                 executor: StepExecutor,
                                 step: StepConfig, 
                                 context: ExecutionContext) -> Any:
        """재시도와 타임아웃을 적용한 실행"""
        
        retry_count = step.retry
        last_error = None
        
        for attempt in range(retry_count + 1):
            try:
                # 타임아웃 적용
                result = await asyncio.wait_for(
                    executor.execute(step, context),
                    timeout=step.timeout
                )
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Step {step.name} timed out (attempt {attempt + 1})")
                last_error = TimeoutError(f"Step timed out after {step.timeout}s")
                
            except Exception as e:
                logger.warning(f"Step {step.name} failed (attempt {attempt + 1}): {e}")
                last_error = e
                
            if attempt < retry_count:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        raise last_error or Exception("Step execution failed")
        
    async def _execute_parallel(self, 
                              step: StepConfig, 
                              context: ExecutionContext) -> List[Any]:
        """병렬 실행"""
        
        branches = step.params.get('branches', [])
        if not branches:
            return []
            
        tasks = []
        for branch_name in branches:
            if branch_name in self.step_executors:
                branch_step = self.step_executors[branch_name]
                task = asyncio.create_task(
                    self._execute_step(branch_step, context)
                )
                tasks.append(task)
                
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 에러 처리
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel branch {branches[i]} failed: {result}")
                
        return results
        
    def _get_next_step(self, 
                      step: StepConfig, 
                      context: ExecutionContext,
                      result: Any = None) -> Optional[str]:
        """다음 실행 단계 결정"""
        
        # 조건부 분기
        if step.type == StepType.CONDITION:
            condition = step.params.get('condition')
            if self._evaluate_condition(condition, context):
                return step.params.get('true_branch')
            else:
                return step.params.get('false_branch')
                
        # 정책 기반 분기
        if step.policy and 'condition' in step.policy:
            policy_condition = step.policy['condition']
            if self._evaluate_condition(policy_condition['if'], context):
                return policy_condition.get('then', {}).get('next')
                
        # 기본 다음 단계
        if step.next_steps:
            return step.next_steps[0]
            
        return None
        
    def _evaluate_condition(self, 
                          condition: Dict[str, Any], 
                          context: ExecutionContext) -> bool:
        """조건 평가"""
        
        if 'raw' in condition:
            # 복잡한 조건은 별도 평가 엔진 사용
            return self._evaluate_raw_condition(condition['raw'], context)
            
        variable = condition.get('variable')
        operator = condition.get('operator')
        value = condition.get('value')
        
        context_value = context.get_variable(variable)
        
        if operator == '<':
            return context_value < value
        elif operator == '>':
            return context_value > value
        elif operator == '==':
            return context_value == value
        elif operator == '!=':
            return context_value != value
        elif operator == '<=':
            return context_value <= value
        elif operator == '>=':
            return context_value >= value
            
        return False
        
    def _evaluate_raw_condition(self, 
                              raw_condition: str, 
                              context: ExecutionContext) -> bool:
        """복잡한 조건 평가"""
        # 간단한 eval 구현 (프로덕션에서는 더 안전한 방법 사용)
        try:
            # 컨텍스트 변수를 로컬 변수로 설정
            local_vars = context.variables.copy()
            return eval(raw_condition, {"__builtins__": {}}, local_vars)
        except Exception as e:
            logger.warning(f"Failed to evaluate condition: {raw_condition}, error: {e}")
            return False
            
    def _get_cache_key(self, 
                      step: StepConfig, 
                      context: ExecutionContext) -> str:
        """캐시 키 생성"""
        # 단계 이름과 주요 컨텍스트 변수로 키 생성
        key_parts = [step.name]
        
        for var_name in ['input_text', 'query', 'intent']:
            if var_name in context.variables:
                key_parts.append(str(context.variables[var_name]))
                
        return ':'.join(key_parts)