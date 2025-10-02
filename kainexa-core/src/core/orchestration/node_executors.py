# src/core/executor/node_executors.py
"""
Node Executors for all node types
모든 노드 타입(Intent, LLM, API, Condition, Loop)의 실행기
"""
import asyncio
import aiohttp
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

from src.core.orchestration.execution_context import (
    ExecutionContext,
    NodeResult,
    NodeStatus,   # ⬅︎ Enum 명시 임포트
)

logger = structlog.get_logger()

# ========== Base Node Executor ==========
class NodeExecutor(ABC):
    """노드 실행기 베이스 클래스"""
    
    @abstractmethod
    async def execute(self, 
                     node: Dict[str, Any],
                     context: ExecutionContext,
                     config: Dict[str, Any]) -> NodeResult:
        """노드 실행"""
        pass
    
    def render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """템플릿 렌더링 (변수 치환)"""
        
        # {{variable}} 형태의 변수 치환
        pattern = r'\{\{(\w+(?:\.\w+)*)\}\}'
        
        def replacer(match):
            key = match.group(1)
            
            # 중첩된 키 지원 (예: order.id)
            keys = key.split('.')
            value = variables
            
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, '')
                else:
                    return match.group(0)  # 치환 실패 시 원본 유지
            
            return str(value)
        
        return re.sub(pattern, replacer, template)
    
    def evaluate_expression(self, expression: str, variables: Dict[str, Any]) -> Any:
        """표현식 평가"""
        
        try:
            # 안전한 평가 환경
            safe_dict = {
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'True': True,
                    'False': False,
                    'None': None,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'round': round
                }
            }
            
            # 간단 헬퍼 (validator에서 'contains(...)' 등을 쓰는 케이스 대비)
            safe_dict.update({
                'contains': lambda container, item: str(item) in str(container),
            })
            safe_dict.update(variables)
            
            return eval(expression, safe_dict)
            
        except Exception as e:
            logger.error(f"Expression evaluation failed: {expression}, error: {e}")
            raise ValueError(f"Invalid expression: {expression}")

# ========== Intent Node Executor ==========
class IntentNodeExecutor(NodeExecutor):
    """의도 분류 노드 실행기"""
    
    async def execute(self,
                     node: Dict[str, Any],
                     context: ExecutionContext,
                     config: Dict[str, Any]) -> NodeResult:
        
        start_time = datetime.utcnow()
        
        # 입력 텍스트 가져오기
        input_text = context.get_variable('input', {}).get('text', '')
        if not input_text:
            input_text = context.get_variable('message', '')
        
        # 의도 분류 수행
        categories = config.get('categories', [])
        confidence_threshold = config.get('confidence_threshold', 0.7)
        fallback_intent = config.get('fallback_intent', 'general')
        
        # 실제 환경에서는 ML 모델 호출
        # 여기서는 간단한 키워드 매칭으로 시뮬레이션
        detected_intent = await self._classify_intent(
            input_text,
            categories,
            confidence_threshold,
            fallback_intent
        )
        
        # 결과 생성
        outputs = {
            'intent': detected_intent['intent'],
            'confidence': detected_intent['confidence'],
            'categories_checked': categories
        }
        
        # 메트릭
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        metrics = {
            'duration_ms': duration_ms,
            'model': config.get('model', 'default-intent-classifier')
        }
        
        logger.info(
            f"Intent classified: {detected_intent['intent']} "
            f"(confidence: {detected_intent['confidence']})"
        )
        
        return NodeResult(
            node_id=node['id'],
            status=NodeStatus.SUCCESS,   # ⬅︎ Enum으로 교체
            outputs=outputs,
            metrics=metrics
        )
    
    async def _classify_intent(self,
                              text: str,
                              categories: List[str],
                              threshold: float,
                              fallback: str) -> Dict[str, Any]:
        """의도 분류 로직 (실제로는 ML 모델 사용)"""
        
        # 키워드 기반 간단한 분류 (데모용)
        intent_keywords = {
            'refund': ['환불', '반품', '취소', 'refund', 'return'],
            'exchange': ['교환', '바꾸', 'exchange', 'swap'],
            'inquiry': ['문의', '질문', '궁금', 'question', 'ask'],
            'complaint': ['불만', '불편', '문제', 'complaint', 'issue'],
            'order': ['주문', '구매', 'order', 'buy', 'purchase']
        }
        
        text_lower = text.lower()
        
        for intent, keywords in intent_keywords.items():
            if intent in categories:
                for keyword in keywords:
                    if keyword in text_lower:
                        return {
                            'intent': intent,
                            'confidence': 0.85
                        }
        
        # Fallback
        return {
            'intent': fallback,
            'confidence': 0.5
        }

# ========== LLM Node Executor ==========  
class LLMNodeExecutor(NodeExecutor):
    """LLM 생성 노드 실행기"""
    
    async def execute(self,
                     node: Dict[str, Any],
                     context: ExecutionContext,
                     config: Dict[str, Any]) -> NodeResult:
        
        start_time = datetime.utcnow()
        
        # 프롬프트 구성
        system_prompt = config.get('system_prompt', '')
        prompt_template = config.get('prompt_template', '')
        
        # 변수 치환
        variables = context.get_all_variables()
        prompt = self.render_template(prompt_template, variables)
        
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # LLM 호출 파라미터
        model = config.get('model', 'solar-10.7b')
        temperature = config.get('temperature', 0.7)
        max_tokens = config.get('max_tokens', 150)
        top_p = config.get('top_p', 0.9)
        
        # 실제 LLM 호출 (또는 시뮬레이션)
        try:
            response = await self._call_llm(
                prompt=full_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                context=context
            )
        except Exception as e:
            # 폴백 처리
            fallback = context.get_policy('fallback.on_llm_error')
            if fallback and fallback.get('action') == 'use_model':
                fallback_model = fallback.get('model', 'gpt2-small')
                logger.info(f"Using fallback model: {fallback_model}")
                
                response = await self._call_llm(
                    prompt=full_prompt,
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    context=context
                )
            else:
                raise
        
        # 결과
        outputs = {
            'response': response['text'],
            'model_used': response['model'],
            'tokens_used': response['tokens']
        }
        
        # 메트릭
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        metrics = {
            'duration_ms': duration_ms,
            'tokens': response['tokens'],
            'model': response['model'],
            'inference_ms': response.get('inference_ms', duration_ms)
        }
        
        return NodeResult(
            node_id=node['id'],
            status=NodeStatus.SUCCESS,   # ⬅︎ Enum으로 교체
            outputs=outputs,
            metrics=metrics
        )
    
    async def _call_llm(self,
                       prompt: str,
                       model: str,
                       temperature: float,
                       max_tokens: int,
                       top_p: float,
                       context: ExecutionContext) -> Dict[str, Any]:
        """LLM API 호출"""
        
        # 실제 Solar LLM 통합
        try:
            from src.models.solar_llm import SolarLLM
            
            # 캐시된 모델 사용
            cache_key = f"llm_model_{model}"
            llm = context.get_cache(cache_key)
            
            if not llm:
                llm = SolarLLM(model_path=model)
                llm.load()
                context.set_cache(cache_key, llm)
            
            # 생성
            response_text = llm.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            
            # 토큰 수 추정 (실제로는 tokenizer 사용)
            tokens = len(response_text.split()) * 1.3
            
            return {
                'text': response_text,
                'model': model,
                'tokens': int(tokens),
                'inference_ms': 100  # 실제 측정값 사용
            }
            
        except ImportError:
            # Solar LLM이 없을 때 시뮬레이션
            logger.warning("Solar LLM not available, using mock response")
            
            mock_responses = {
                'refund': "네, 고객님의 환불 요청을 확인했습니다. 주문번호를 확인 후 환불 절차를 진행하겠습니다.",
                'exchange': "제품 교환을 도와드리겠습니다. 교환 사유와 원하시는 제품을 알려주시면 처리해드리겠습니다.",
                'general': "안녕하세요! 무엇을 도와드릴까요? 궁금하신 점이 있으시면 편하게 말씀해주세요."
            }
            
            # 의도 기반 응답 선택
            intent = context.get_variable('intent', 'general')
            response_text = mock_responses.get(
                intent,
                mock_responses['general']
            )
            
            return {
                'text': response_text,
                'model': f"{model}-mock",
                'tokens': len(response_text.split()),
                'inference_ms': 50
            }

# ========== API Node Executor ==========
class APINodeExecutor(NodeExecutor):
    """외부 API 호출 노드 실행기"""
    
    async def execute(self,
                     node: Dict[str, Any],
                     context: ExecutionContext,
                     config: Dict[str, Any]) -> NodeResult:
        
        start_time = datetime.utcnow()
        
        # API 설정
        method = config.get('method', 'GET')
        url_template = config.get('url', '')
        headers = config.get('headers', {})
        timeout = config.get('timeout', 5000) / 1000.0  # ms to seconds
        retry_config = config.get('retry', {})
        
        # URL과 헤더 렌더링 (변수 치환)
        variables = context.get_all_variables()
        url = self.render_template(url_template, variables)
        
        rendered_headers = {}
        for key, value in headers.items():
            rendered_headers[key] = self.render_template(str(value), variables)
        
        # 요청 바디 준비
        body = None
        if method in ['POST', 'PUT', 'PATCH']:
            body_template = config.get('body', {})
            if isinstance(body_template, str):
                body = self.render_template(body_template, variables)
            else:
                body = json.dumps(body_template)
            # JSON 바디인데 헤더에 Content-Type 없으면 기본 지정
            if 'Content-Type' not in rendered_headers:
                rendered_headers['Content-Type'] = 'application/json'
   
        
        # API 호출 (재시도 로직 포함)
        max_attempts = retry_config.get('max_attempts', 3)
        backoff = retry_config.get('backoff', 'exponential')
        
        response_data = None
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response_data = await self._call_api(
                    method=method,
                    url=url,
                    headers=rendered_headers,
                    body=body,
                    timeout=timeout
                )
                break
                
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    # 재시도 대기
                    if backoff == 'exponential':
                        wait_time = 2 ** attempt
                    else:
                        wait_time = 1
                    
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
        
        if response_data is None:
            # 폴백 처리
            fallback = context.get_policy('fallback.on_api_error')
            if fallback and fallback.get('action') == 'use_cache':
                cache_key = f"api_cache_{url}_{method}"
                cached = context.get_cache(cache_key)
                
                if cached:
                    logger.info("Using cached API response")
                    response_data = cached
                else:
                    raise last_error
            else:
                raise last_error
        else:
            # 캐시 저장
            cache_key = f"api_cache_{url}_{method}"
            context.set_cache(cache_key, response_data)
        
        # 응답 매핑
        response_mapping = config.get('response_mapping', {})
        outputs = {}
        
        if response_mapping:
            for output_key, json_path in response_mapping.items():
                value = self._extract_json_path(response_data, json_path)
                outputs[output_key] = value
        else:
            outputs['response'] = response_data
        
        # 메트릭
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        metrics = {
            'duration_ms': duration_ms,
            'api_calls': 1,
            'status_code': response_data.get('_status_code', 200)
        }
        
        return NodeResult(
            node_id=node['id'],
            status=NodeStatus.SUCCESS,   # ⬅︎ Enum으로 교체
            outputs=outputs,
            metrics=metrics
        )
    
    async def _call_api(self,
                       method: str,
                       url: str,
                       headers: Dict[str, str],
                       body: Optional[str],
                       timeout: float) -> Dict[str, Any]:
        """실제 API 호출"""
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    # 일부 API가 JSON이 아닐 수도 있으므로 폴백 처리
                    try:
                        response_data = await response.json(content_type=None)
                    except Exception:
                        text_body = await response.text()
                        response_data = {'_raw': text_body}
                    
                    response_data['_status_code'] = response.status
                    
                    if response.status >= 400:
                        raise aiohttp.ClientError(
                            f"API returned status {response.status}"
                        )
                    
                    return response_data
                    
            except asyncio.TimeoutError:
                raise TimeoutError(f"API call timeout after {timeout}s")
            except aiohttp.ClientError as e:
                raise RuntimeError(f"API call failed: {e}")
            except Exception as e:
                # 개발 모드에서는 모의 응답
                if 'localhost' in url or '127.0.0.1' in url:
                    logger.warning(f"Using mock API response for {url}")
                    return {
                        '_status_code': 200,
                        'mock': True,
                        'data': {
                            'order_id': 'A-123456',
                            'status': 'completed',
                            'amount': 50000
                        }
                    }
                raise
    
    def _extract_json_path(self, data: Dict[str, Any], path: str) -> Any:
        """JSONPath 표현식으로 값 추출"""
        
        # 간단한 JSONPath 구현 ($.data.status 형태)
        if path.startswith('$'):
            path = path[1:]
        
        if path.startswith('.'):
            path = path[1:]
        
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None
        
        return value

# ========== Condition Node Executor ==========
class ConditionNodeExecutor(NodeExecutor):
    """조건 분기 노드 실행기"""
    
    async def execute(self,
                     node: Dict[str, Any],
                     context: ExecutionContext,
                     config: Dict[str, Any]) -> NodeResult:
        
        start_time = datetime.utcnow()
        
        # 조건 표현식
        expression = config.get('expression', '')
        variables_needed = config.get('variables', [])
        
        # 필요한 변수 수집
        variables = {}
        for var in variables_needed:
            variables[var] = context.get_variable(var)
        
        # 추가로 모든 변수 포함
        all_vars = context.get_all_variables()
        variables.update(all_vars)
        
        # 조건 평가
        try:
            result = self.evaluate_expression(expression, variables)
            condition_met = bool(result)
        except Exception as e:
            logger.error(f"Condition evaluation failed: {expression}, error: {e}")
            condition_met = False
        
        # 결과
        outputs = {
            'condition_met': condition_met,
            'expression': expression,
            'evaluated_value': str(result) if 'result' in locals() else 'error'
        }
        
        # 메트릭
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        metrics = {
            'duration_ms': duration_ms
        }
        
        logger.info(
            f"Condition evaluated: {expression} = {condition_met}"
        )
        
        return NodeResult(
            node_id=node['id'],
            status=NodeStatus.SUCCESS,   # ⬅︎ Enum으로 교체
            outputs=outputs,
            metrics=metrics
        )

# ========== Loop Node Executor ==========
class LoopNodeExecutor(NodeExecutor):
    """반복 실행 노드 실행기"""
    
    async def execute(self,
                     node: Dict[str, Any],
                     context: ExecutionContext,
                     config: Dict[str, Any]) -> NodeResult:
        
        start_time = datetime.utcnow()
        
        # 반복 설정
        iterator_var = config.get('iterator', 'items')
        max_iterations = config.get('max_iterations', 10)
        parallel = config.get('parallel', False)
        body_nodes = config.get('body', [])
        
        # 반복할 데이터 가져오기
        items = context.get_variable(iterator_var, [])
        if not isinstance(items, (list, tuple)):
            items = [items]  # 단일 항목도 리스트로 변환
        
        # 반복 횟수 제한
        items = items[:max_iterations]
        
        # 반복 실행 결과
        loop_results = []
        
        if parallel:
            # 병렬 실행
            tasks = []
            for idx, item in enumerate(items):
                # 각 반복을 위한 독립적인 컨텍스트 (clone 없을 때 대비)
                if hasattr(context, "clone") and callable(getattr(context, "clone")):
                    loop_context = context.clone()
                else:
                    # 최소 폴백: 변수/정책만 얕게 복사한 새 컨텍스트
                    loop_context = ExecutionContext()
                    loop_context.variables.update(context.get_all_variables())
                    loop_context.policies.update(getattr(context, "policies", {}))
                
                loop_context.set_variable('loop_index', idx)
                loop_context.set_variable('loop_item', item)
                
                task = self._execute_loop_iteration(
                    idx, item, body_nodes, loop_context
                )
                tasks.append(task)
            
            loop_results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # 순차 실행
            for idx, item in enumerate(items):
                context.set_variable('loop_index', idx)
                context.set_variable('loop_item', item)
                
                result = await self._execute_loop_iteration(
                    idx, item, body_nodes, context
                )
                loop_results.append(result)
                
                # 중단 조건 체크
                if context.get_variable('loop_break', False):
                    break
        
        # 결과 정리
        outputs = {
            'iterations': len(loop_results),
            'results': loop_results,
            'completed': True
        }
        
        # 메트릭
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        metrics = {
            'duration_ms': duration_ms,
            'iterations': len(loop_results)
        }
        
        return NodeResult(
            node_id=node['id'],
            status=NodeStatus.SUCCESS,   # ⬅︎ Enum으로 교체
            outputs=outputs,
            metrics=metrics
        )
    
    async def _execute_loop_iteration(self,
                                     index: int,
                                     item: Any,
                                     body_nodes: List[str],
                                     context: ExecutionContext) -> Dict[str, Any]:
        """단일 반복 실행"""
        
        iteration_result = {
            'index': index,
            'item': item,
            'outputs': {}
        }
        
        # body에 정의된 노드들 실행
        # (실제로는 GraphExecutor를 통해 실행해야 함)
        for node_id in body_nodes:
            # 간단한 시뮬레이션
            iteration_result['outputs'][node_id] = {
                'processed': True,
                'item': item
            }
        
        return iteration_result

# ========== Node Executor Factory ==========
_node_executors = {
    'intent': IntentNodeExecutor(),
    'llm': LLMNodeExecutor(),
    'api': APINodeExecutor(),
    'condition': ConditionNodeExecutor(),
    'loop': LoopNodeExecutor()
}

def get_node_executor(node_type: str) -> Optional[NodeExecutor]:
    """노드 타입에 맞는 실행기 반환"""
    return _node_executors.get(node_type)

def register_node_executor(node_type: str, executor: NodeExecutor):
    """커스텀 노드 실행기 등록"""
    _node_executors[node_type] = executor