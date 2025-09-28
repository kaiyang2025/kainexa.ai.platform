# src/orchestration/policy_engine.py
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
from collections import defaultdict
import structlog

logger = structlog.get_logger()

class PolicyAction(Enum):
    """정책 액션"""
    CONTINUE = "continue"
    SKIP = "skip"
    RETRY = "retry"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    CACHE = "cache"
    THROTTLE = "throttle"

@dataclass
class PolicyDecision:
    """정책 결정"""
    action: PolicyAction
    reason: str
    metadata: Dict[str, Any] = None

class PolicyEngine:
    """정책 엔진 - 실행 정책 관리 및 평가"""
    
    def __init__(self):
        self.rate_limiters = defaultdict(RateLimiter)
        self.cost_tracker = CostTracker()
        self.confidence_thresholds = {}
        self.fallback_handlers = {}
        
    async def evaluate(self, 
                      step, 
                      context, 
                      global_policies: Dict[str, Any]) -> PolicyDecision:
        """정책 평가"""
        
        # 1. Rate Limiting 확인
        rate_limit_decision = await self._check_rate_limit(step, context)
        if rate_limit_decision.action != PolicyAction.CONTINUE:
            return rate_limit_decision
            
        # 2. 비용 제한 확인
        cost_decision = await self._check_cost_limit(step, context, global_policies)
        if cost_decision.action != PolicyAction.CONTINUE:
            return cost_decision
            
        # 3. 신뢰도 임계값 확인
        confidence_decision = self._check_confidence_threshold(step, context)
        if confidence_decision.action != PolicyAction.CONTINUE:
            return confidence_decision
            
        # 4. 타임아웃 정책
        timeout_decision = self._check_timeout_policy(step, context)
        if timeout_decision.action != PolicyAction.CONTINUE:
            return timeout_decision
            
        # 5. 커스텀 정책
        if step.policy:
            custom_decision = await self._evaluate_custom_policy(step.policy, context)
            if custom_decision.action != PolicyAction.CONTINUE:
                return custom_decision
                
        return PolicyDecision(
            action=PolicyAction.CONTINUE,
            reason="All policies passed"
        )
        
    async def _check_rate_limit(self, step, context) -> PolicyDecision:
        """Rate Limiting 확인"""
        
        if not hasattr(step, 'params') or 'rate_limit' not in step.params:
            return PolicyDecision(PolicyAction.CONTINUE, "No rate limit")
            
        rate_limit = step.params['rate_limit']
        limiter_key = f"{step.name}:{context.session_id}"
        limiter = self.rate_limiters[limiter_key]
        
        if not limiter.allow_request(rate_limit['requests'], rate_limit['period']):
            logger.warning(f"Rate limit exceeded for {step.name}")
            
            # Throttle 시간 계산
            wait_time = limiter.get_wait_time()
            
            return PolicyDecision(
                action=PolicyAction.THROTTLE,
                reason=f"Rate limit exceeded, wait {wait_time}s",
                metadata={'wait_time': wait_time}
            )
            
        return PolicyDecision(PolicyAction.CONTINUE, "Rate limit OK")
        
    async def _check_cost_limit(self, 
                               step, 
                               context,
                               global_policies: Dict[str, Any]) -> PolicyDecision:
        """비용 제한 확인"""
        
        # 글로벌 비용 정책
        if 'cost_limit' in global_policies:
            session_cost = self.cost_tracker.get_session_cost(context.session_id)
            max_cost = global_policies['cost_limit'].get('max_per_session', float('inf'))
            
            if session_cost >= max_cost:
                logger.warning(f"Cost limit exceeded for session {context.session_id}")
                
                # Fallback 모델로 전환
                if 'fallback_model' in global_policies['cost_limit']:
                    return PolicyDecision(
                        action=PolicyAction.FALLBACK,
                        reason="Cost limit exceeded, using fallback model",
                        metadata={'fallback_model': global_policies['cost_limit']['fallback_model']}
                    )
                    
                return PolicyDecision(
                    action=PolicyAction.SKIP,
                    reason="Cost limit exceeded"
                )
                
        # 단계별 비용 예측
        if step.type == 'llm_generate':
            estimated_cost = self.cost_tracker.estimate_llm_cost(
                model=step.params.get('model', 'default'),
                max_tokens=step.params.get('max_tokens', 512)
            )
            
            # 예상 비용이 임계값을 초과하면 경고
            if estimated_cost > 1.0:  # $1 이상
                logger.info(f"High cost operation: ${estimated_cost:.2f}")
                
        return PolicyDecision(PolicyAction.CONTINUE, "Cost limit OK")
        
    def _check_confidence_threshold(self, step, context) -> PolicyDecision:
        """신뢰도 임계값 확인"""
        
        if not step.policy or 'confidence_threshold' not in step.policy:
            return PolicyDecision(PolicyAction.CONTINUE, "No confidence threshold")
            
        threshold = step.policy['confidence_threshold']
        confidence = context.get_variable('confidence', 1.0)
        
        if confidence < threshold:
            logger.info(f"Confidence {confidence} below threshold {threshold}")
            
            # Escalation 정책
            if 'escalate_on_low_confidence' in step.policy:
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason=f"Low confidence: {confidence}",
                    metadata={'confidence': confidence}
                )
                
            # Fallback 정책
            if 'fallback_on_low_confidence' in step.policy:
                return PolicyDecision(
                    action=PolicyAction.FALLBACK,
                    reason=f"Low confidence: {confidence}",
                    metadata={'fallback_to': step.policy['fallback_on_low_confidence']}
                )
                
        return PolicyDecision(PolicyAction.CONTINUE, "Confidence OK")
        
    def _check_timeout_policy(self, step, context) -> PolicyDecision:
        """타임아웃 정책 확인"""
        
        if not hasattr(step, 'timeout'):
            return PolicyDecision(PolicyAction.CONTINUE, "No timeout policy")
            
        # 전체 실행 시간 확인
        total_time = time.time() - context.start_time
        max_total_time = 60.0  # 60초 제한
        
        if total_time > max_total_time:
            logger.warning(f"Total execution time {total_time}s exceeded limit")
            
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason=f"Total timeout: {total_time}s",
                metadata={'total_time': total_time}
            )
            
        return PolicyDecision(PolicyAction.CONTINUE, "Timeout OK")
        
    async def _evaluate_custom_policy(self, 
                                     policy: Dict[str, Any],
                                     context) -> PolicyDecision:
        """커스텀 정책 평가"""
        
        # 조건부 정책
        if 'condition' in policy:
            condition = policy['condition']
            
            # 조건 평가 (간단한 구현)
            if self._evaluate_policy_condition(condition, context):
                action_config = condition.get('then', {})
            else:
                action_config = condition.get('else', {})
                
            # 액션 결정
            if 'action' in action_config:
                action_str = action_config['action']
                try:
                    action = PolicyAction(action_str)
                    return PolicyDecision(
                        action=action,
                        reason=f"Custom policy: {action_str}",
                        metadata=action_config.get('metadata', {})
                    )
                except ValueError:
                    logger.warning(f"Unknown policy action: {action_str}")
                    
        # Retry 정책
        if 'retry' in policy:
            retry_config = policy['retry']
            max_retries = retry_config.get('max', 3)
            
            # 현재 재시도 횟수 확인
            retry_count = context.get_variable('retry_count', 0)
            
            if retry_count >= max_retries:
                return PolicyDecision(
                    action=PolicyAction.FALLBACK,
                    reason=f"Max retries ({max_retries}) exceeded"
                )
                
        # A/B 테스트 정책
        if 'ab_test' in policy:
            ab_config = policy['ab_test']
            variant = self._select_ab_variant(ab_config, context)
            
            context.set_variable('ab_variant', variant)
            logger.info(f"A/B test variant selected: {variant}")
            
        return PolicyDecision(PolicyAction.CONTINUE, "Custom policy OK")
        
    def _evaluate_policy_condition(self, 
                                  condition: Dict[str, Any],
                                  context) -> bool:
        """정책 조건 평가"""
        
        if 'variable' in condition:
            var_name = condition['variable']
            operator = condition.get('operator', '==')
            value = condition['value']
            
            context_value = context.get_variable(var_name)
            
            if operator == '<':
                return context_value < value
            elif operator == '>':
                return context_value > value
            elif operator == '==':
                return context_value == value
            elif operator == '!=':
                return context_value != value
            elif operator == 'in':
                return context_value in value
            elif operator == 'not_in':
                return context_value not in value
                
        return False
        
    def _select_ab_variant(self, 
                          ab_config: Dict[str, Any],
                          context) -> str:
        """A/B 테스트 변형 선택"""
        import hashlib
        
        # 세션 기반 일관된 변형 선택
        session_hash = hashlib.md5(context.session_id.encode()).hexdigest()
        hash_int = int(session_hash[:8], 16)
        
        # 가중치 기반 선택
        variants = ab_config.get('variants', [])
        total_weight = sum(v.get('weight', 1) for v in variants)
        
        threshold = hash_int % total_weight
        current = 0
        
        for variant in variants:
            current += variant.get('weight', 1)
            if threshold < current:
                return variant['name']
                
        return variants[0]['name'] if variants else 'control'


class RateLimiter:
    """Rate Limiter 구현"""
    
    def __init__(self):
        self.requests = []
        
    def allow_request(self, max_requests: int, period: float) -> bool:
        """요청 허용 여부 확인"""
        now = time.time()
        
        # 만료된 요청 제거
        self.requests = [t for t in self.requests if now - t < period]
        
        # 제한 확인
        if len(self.requests) >= max_requests:
            return False
            
        # 요청 기록
        self.requests.append(now)
        return True
        
    def get_wait_time(self) -> float:
        """다음 요청까지 대기 시간"""
        if not self.requests:
            return 0
            
        oldest = min(self.requests)
        return max(0, 60 - (time.time() - oldest))  # 60초 기준


class CostTracker:
    """비용 추적"""
    
    def __init__(self):
        self.session_costs = defaultdict(float)
        self.model_costs = {
            'solar-10.7b': 0.001,  # per 1K tokens
            'gpt-4': 0.03,
            'gpt-3.5': 0.002,
            'default': 0.001
        }
        
    def get_session_cost(self, session_id: str) -> float:
        """세션 비용 조회"""
        return self.session_costs[session_id]
        
    def add_cost(self, session_id: str, cost: float):
        """비용 추가"""
        self.session_costs[session_id] += cost
        
    def estimate_llm_cost(self, model: str, max_tokens: int) -> float:
        """LLM 비용 예측"""
        cost_per_1k = self.model_costs.get(model, 0.001)
        return (max_tokens / 1000) * cost_per_1k


class FallbackHandler:
    """Fallback 처리"""
    
    def __init__(self):
        self.fallback_chains = {}
        
    def register_fallback(self, 
                         primary: str,
                         fallbacks: List[str]):
        """Fallback 체인 등록"""
        self.fallback_chains[primary] = fallbacks
        
    def get_fallback(self, primary: str, attempt: int = 0) -> Optional[str]:
        """Fallback 옵션 조회"""
        if primary not in self.fallback_chains:
            return None
            
        fallbacks = self.fallback_chains[primary]
        
        if attempt < len(fallbacks):
            return fallbacks[attempt]
            
        return None


class CircuitBreaker:
    """Circuit Breaker 패턴 구현"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = defaultdict(list)
        self.last_failure_time = {}
        
    def is_open(self, service: str) -> bool:
        """서킷 브레이커 상태 확인"""
        
        # 복구 시간 확인
        if service in self.last_failure_time:
            if time.time() - self.last_failure_time[service] > self.recovery_timeout:
                # Reset
                self.failures[service] = []
                del self.last_failure_time[service]
                return False
                
        # 실패 횟수 확인
        recent_failures = self._get_recent_failures(service)
        
        if len(recent_failures) >= self.failure_threshold:
            return True
            
        return False
        
    def record_failure(self, service: str):
        """실패 기록"""
        now = time.time()
        self.failures[service].append(now)
        self.last_failure_time[service] = now
        
        # 오래된 기록 정리
        self.failures[service] = self._get_recent_failures(service)
        
    def record_success(self, service: str):
        """성공 기록"""
        if service in self.failures:
            self.failures[service] = []
        if service in self.last_failure_time:
            del self.last_failure_time[service]
            
    def _get_recent_failures(self, service: str) -> List[float]:
        """최근 실패 기록 조회"""
        now = time.time()
        window = 60.0  # 60초 윈도우
        
        return [
            f for f in self.failures[service]
            if now - f < window
        ]