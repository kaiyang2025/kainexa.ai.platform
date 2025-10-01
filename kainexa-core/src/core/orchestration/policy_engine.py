# src/core/orchestration/policy_engine.py
"""
Kainexa Policy Engine - 완전한 구현
실행 정책 관리, 평가, SLA/비용/에스컬레이션 처리
"""
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from collections import defaultdict
import structlog
import hashlib
from datetime import datetime, timedelta
import json

logger = structlog.get_logger()

# ========== Policy Actions ==========
class PolicyAction(Enum):
    """정책 액션"""
    CONTINUE = "continue"           # 계속 진행
    SKIP = "skip"                  # 단계 건너뛰기
    RETRY = "retry"                # 재시도
    FALLBACK = "fallback"          # 폴백 실행
    ESCALATE = "escalate"          # 상담사 에스컬레이션
    CACHE = "cache"                # 캐시 사용
    THROTTLE = "throttle"          # 속도 제한
    TERMINATE = "terminate"        # 종료
    FORK = "fork"                  # 분기 실행

# ========== Policy Decision ==========
@dataclass
class PolicyDecision:
    """정책 결정"""
    action: PolicyAction
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 높을수록 우선
    
    def __str__(self):
        return f"PolicyDecision(action={self.action.value}, reason={self.reason})"

# ========== Policy Condition ==========
@dataclass
class PolicyCondition:
    """정책 조건"""
    field: str
    operator: str  # ==, !=, <, >, <=, >=, in, not_in, regex, exists
    value: Any
    combine: str = "AND"  # AND, OR
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """조건 평가"""
        # context에서 값 추출 (nested path 지원)
        actual_value = self._get_nested_value(context, self.field)
        
        # 연산자별 평가
        if self.operator == '==':
            return actual_value == self.value
        elif self.operator == '!=':
            return actual_value != self.value
        elif self.operator == '<':
            return actual_value < self.value
        elif self.operator == '>':
            return actual_value > self.value
        elif self.operator == '<=':
            return actual_value <= self.value
        elif self.operator == '>=':
            return actual_value >= self.value
        elif self.operator == 'in':
            return actual_value in self.value
        elif self.operator == 'not_in':
            return actual_value not in self.value
        elif self.operator == 'regex':
            import re
            return bool(re.match(self.value, str(actual_value)))
        elif self.operator == 'exists':
            return actual_value is not None
        else:
            logger.warning(f"Unknown operator: {self.operator}")
            return False
    
    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Nested path에서 값 추출 (예: user.profile.age)"""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

# ========== Rate Limiter ==========
class RateLimiter:
    """Rate Limiter 구현"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.locks = defaultdict(asyncio.Lock)
    
    async def check_limit(self, 
                         key: str, 
                         max_requests: int, 
                         period_seconds: float) -> Tuple[bool, float]:
        """
        Rate limit 확인
        Returns: (허용여부, 대기시간)
        """
        async with self.locks[key]:
            now = time.time()
            
            # 만료된 요청 제거
            self.requests[key] = [
                t for t in self.requests[key] 
                if now - t < period_seconds
            ]
            
            # 제한 확인
            if len(self.requests[key]) >= max_requests:
                # 대기시간 계산
                oldest = min(self.requests[key])
                wait_time = period_seconds - (now - oldest)
                return False, wait_time
            
            # 요청 기록
            self.requests[key].append(now)
            return True, 0
    
    def reset(self, key: str):
        """특정 키의 rate limit 리셋"""
        if key in self.requests:
            del self.requests[key]

# ========== Cost Tracker ==========
class CostTracker:
    """비용 추적 및 관리"""
    
    def __init__(self, cost_config: Optional[Dict[str, float]] = None):
        self.session_costs = defaultdict(float)
        self.model_costs = cost_config or {
            # 모델별 1K 토큰당 비용 (USD)
            'solar-10.7b': 0.001,
            'gpt-4': 0.03,
            'gpt-3.5': 0.002,
            'claude-3': 0.01,
            'gemini-pro': 0.005,
            'slm-ko-3b': 0.0005,  # 경량모델
            'default': 0.001
        }
        self.daily_costs = defaultdict(lambda: defaultdict(float))
        self.monthly_costs = defaultdict(lambda: defaultdict(float))
        
    def get_session_cost(self, session_id: str) -> float:
        """세션 누적 비용 조회"""
        return self.session_costs[session_id]
    
    def add_cost(self, 
                 session_id: str, 
                 cost: float,
                 tenant_id: Optional[str] = None):
        """비용 추가"""
        self.session_costs[session_id] += cost
        
        # 일별/월별 집계
        today = datetime.now().strftime('%Y-%m-%d')
        month = datetime.now().strftime('%Y-%m')
        
        if tenant_id:
            self.daily_costs[tenant_id][today] += cost
            self.monthly_costs[tenant_id][month] += cost
    
    def estimate_llm_cost(self, 
                         model: str, 
                         tokens: int,
                         is_input: bool = True) -> float:
        """LLM 비용 예측"""
        cost_per_1k = self.model_costs.get(model, self.model_costs['default'])
        
        # Output은 보통 Input보다 2배 비용
        if not is_input:
            cost_per_1k *= 2
            
        return (tokens / 1000) * cost_per_1k
    
    def check_budget(self, 
                     tenant_id: str,
                     budget_type: str = 'monthly',
                     limit: float = float('inf')) -> Tuple[bool, float]:
        """예산 확인"""
        if budget_type == 'daily':
            today = datetime.now().strftime('%Y-%m-%d')
            current = self.daily_costs[tenant_id][today]
        else:  # monthly
            month = datetime.now().strftime('%Y-%m')
            current = self.monthly_costs[tenant_id][month]
        
        return current < limit, limit - current

# ========== Fallback Handler ==========
class FallbackHandler:
    """Fallback 처리"""
    
    def __init__(self):
        self.fallback_chains = {}
        self.fallback_history = defaultdict(list)
        
    def register_fallback(self, 
                         primary: str,
                         fallbacks: List[Dict[str, Any]]):
        """
        Fallback 체인 등록
        fallbacks: [{'target': 'model_name', 'condition': {...}}]
        """
        self.fallback_chains[primary] = fallbacks
    
    def get_fallback(self, 
                     primary: str,
                     context: Dict[str, Any]) -> Optional[str]:
        """컨텍스트 기반 폴백 선택"""
        if primary not in self.fallback_chains:
            return None
            
        for fallback in self.fallback_chains[primary]:
            # 조건이 있으면 평가
            if 'condition' in fallback:
                condition = PolicyCondition(**fallback['condition'])
                if condition.evaluate(context):
                    return fallback['target']
            else:
                # 조건 없으면 첫 번째 폴백 사용
                return fallback['target']
                
        return None
    
    def record_fallback(self, 
                       session_id: str,
                       from_model: str,
                       to_model: str,
                       reason: str):
        """Fallback 이력 기록"""
        self.fallback_history[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'from': from_model,
            'to': to_model,
            'reason': reason
        })

# ========== Escalation Manager ==========
class EscalationManager:
    """에스컬레이션 관리"""
    
    def __init__(self):
        self.escalation_rules = {}
        self.escalation_queue = defaultdict(list)
        self.agent_pools = defaultdict(list)
        
    def register_rule(self, 
                     name: str,
                     conditions: List[PolicyCondition],
                     target: str,  # agent_pool, supervisor, expert
                     priority: int = 0):
        """에스컬레이션 규칙 등록"""
        self.escalation_rules[name] = {
            'conditions': conditions,
            'target': target,
            'priority': priority
        }
    
    async def check_escalation(self, 
                              context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """에스컬레이션 필요 여부 확인"""
        triggered_rules = []
        
        for name, rule in self.escalation_rules.items():
            # 모든 조건 평가 (AND 연산)
            if all(cond.evaluate(context) for cond in rule['conditions']):
                triggered_rules.append((rule['priority'], name, rule))
        
        if triggered_rules:
            # 우선순위 순으로 정렬
            triggered_rules.sort(reverse=True, key=lambda x: x[0])
            _, name, rule = triggered_rules[0]
            
            return {
                'rule_name': name,
                'target': rule['target'],
                'priority': rule['priority']
            }
        
        return None
    
    async def escalate(self, 
                       session_id: str,
                       target: str,
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """에스컬레이션 실행"""
        escalation_request = {
            'session_id': session_id,
            'target': target,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        # 큐에 추가
        self.escalation_queue[target].append(escalation_request)
        
        # 가용 상담사 확인 및 할당
        if target in self.agent_pools and self.agent_pools[target]:
            agent = self.agent_pools[target].pop(0)
            escalation_request['assigned_to'] = agent
            escalation_request['status'] = 'assigned'
            
        logger.info(f"Escalated session {session_id} to {target}")
        
        return escalation_request

# ========== Main Policy Engine ==========
class PolicyEngine:
    """정책 엔진 - 모든 정책 관리 및 평가"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rate_limiter = RateLimiter()
        self.cost_tracker = CostTracker(
            self.config.get('cost_config')
        )
        self.fallback_handler = FallbackHandler()
        self.escalation_manager = EscalationManager()
        
        # 정책 캐시
        self.policy_cache = {}
        
        # 커스텀 정책 핸들러
        self.custom_handlers: Dict[str, Callable] = {}
        
        # 정책 실행 통계
        self.stats = defaultdict(lambda: defaultdict(int))
        
    async def evaluate(self, 
                      step: Any,
                      context: Dict[str, Any],
                      global_policies: Optional[Dict[str, Any]] = None) -> PolicyDecision:
        """
        종합 정책 평가
        우선순위: 
        1. Rate Limiting
        2. 비용 제한
        3. SLA/타임아웃
        4. 신뢰도 임계값
        5. 커스텀 정책
        """
        
        decisions = []
        
        # 1. Rate Limiting 확인
        if hasattr(step, 'rate_limit'):
            decision = await self._check_rate_limit(step, context)
            if decision.action != PolicyAction.CONTINUE:
                decisions.append((decision.priority, decision))
        
        # 2. 비용 제한 확인
        decision = await self._check_cost_limit(step, context, global_policies)
        if decision.action != PolicyAction.CONTINUE:
            decisions.append((decision.priority, decision))
        
        # 3. SLA/타임아웃 확인
        decision = self._check_sla_timeout(step, context, global_policies)
        if decision.action != PolicyAction.CONTINUE:
            decisions.append((decision.priority, decision))
        
        # 4. 신뢰도 확인
        decision = self._check_confidence(step, context)
        if decision.action != PolicyAction.CONTINUE:
            decisions.append((decision.priority, decision))
        
        # 5. 감정 분석 기반 정책
        decision = await self._check_sentiment_policy(step, context)
        if decision.action != PolicyAction.CONTINUE:
            decisions.append((decision.priority, decision))
        
        # 6. 커스텀 정책
        if hasattr(step, 'policy') and step.policy:
            decision = await self._evaluate_custom_policy(step.policy, context)
            if decision.action != PolicyAction.CONTINUE:
                decisions.append((decision.priority, decision))
        
        # 우선순위 기반 결정
        if decisions:
            decisions.sort(reverse=True, key=lambda x: x[0])
            final_decision = decisions[0][1]
            
            # 통계 기록
            self._record_stats(step.name if hasattr(step, 'name') else 'unknown', 
                             final_decision)
            
            return final_decision
        
        return PolicyDecision(
            action=PolicyAction.CONTINUE,
            reason="All policies passed"
        )
    
    async def _check_rate_limit(self, step: Any, context: Dict[str, Any]) -> PolicyDecision:
        """Rate Limiting 확인"""
        rate_config = step.rate_limit
        
        # 키 생성 (세션별 또는 전역)
        scope = rate_config.get('scope', 'session')
        if scope == 'global':
            limiter_key = f"{step.name}:global"
        else:
            limiter_key = f"{step.name}:{context.get('session_id', 'unknown')}"
        
        # Rate limit 확인
        allowed, wait_time = await self.rate_limiter.check_limit(
            limiter_key,
            rate_config['max_requests'],
            rate_config['period_seconds']
        )
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {limiter_key}, wait {wait_time:.1f}s")
            
            return PolicyDecision(
                action=PolicyAction.THROTTLE,
                reason=f"Rate limit exceeded, wait {wait_time:.1f}s",
                metadata={'wait_time': wait_time, 'key': limiter_key},
                priority=90  # 높은 우선순위
            )
        
        return PolicyDecision(PolicyAction.CONTINUE, "Rate limit OK")
    
    async def _check_cost_limit(self, 
                               step: Any,
                               context: Dict[str, Any],
                               global_policies: Optional[Dict[str, Any]]) -> PolicyDecision:
        """비용 제한 확인"""
        
        session_id = context.get('session_id', 'unknown')
        tenant_id = context.get('tenant_id', 'default')
        
        # 글로벌 비용 정책
        if global_policies and 'cost_limit' in global_policies:
            cost_config = global_policies['cost_limit']
            
            # 세션 비용 확인
            session_cost = self.cost_tracker.get_session_cost(session_id)
            max_session = cost_config.get('max_per_session', float('inf'))
            
            if session_cost >= max_session:
                logger.warning(f"Session cost limit exceeded: ${session_cost:.4f}")
                
                # Fallback 모델 확인
                if 'fallback_model' in cost_config:
                    return PolicyDecision(
                        action=PolicyAction.FALLBACK,
                        reason=f"Cost limit exceeded (${session_cost:.4f})",
                        metadata={
                            'fallback_model': cost_config['fallback_model'],
                            'current_cost': session_cost
                        },
                        priority=80
                    )
                
                return PolicyDecision(
                    action=PolicyAction.TERMINATE,
                    reason=f"Cost limit exceeded (${session_cost:.4f})",
                    metadata={'current_cost': session_cost},
                    priority=85
                )
            
            # 월별 예산 확인
            if 'monthly_budget' in cost_config:
                within_budget, remaining = self.cost_tracker.check_budget(
                    tenant_id,
                    'monthly',
                    cost_config['monthly_budget']
                )
                
                if not within_budget:
                    return PolicyDecision(
                        action=PolicyAction.TERMINATE,
                        reason="Monthly budget exceeded",
                        metadata={'remaining_budget': 0},
                        priority=85
                    )
                
                # 예산 80% 경고
                if remaining < cost_config['monthly_budget'] * 0.2:
                    logger.warning(f"Monthly budget warning: ${remaining:.2f} remaining")
        
        # 단계별 비용 예측 (LLM 노드)
        if hasattr(step, 'type') and step.type == 'llm':
            model = step.params.get('model', 'default')
            max_tokens = step.params.get('max_tokens', 512)
            
            estimated_cost = self.cost_tracker.estimate_llm_cost(
                model, max_tokens, is_input=False
            )
            
            # 고비용 경고
            if estimated_cost > 0.1:  # $0.10 이상
                logger.info(f"High cost operation: ${estimated_cost:.4f} for {model}")
                
                # 대체 모델 제안
                if estimated_cost > 0.5:  # $0.50 이상
                    return PolicyDecision(
                        action=PolicyAction.FALLBACK,
                        reason=f"High cost operation (${estimated_cost:.4f})",
                        metadata={
                            'estimated_cost': estimated_cost,
                            'fallback_model': 'slm-ko-3b'
                        },
                        priority=70
                    )
        
        return PolicyDecision(PolicyAction.CONTINUE, "Cost limit OK")
    
    def _check_sla_timeout(self, 
                          step: Any,
                          context: Dict[str, Any],
                          global_policies: Optional[Dict[str, Any]]) -> PolicyDecision:
        """SLA 및 타임아웃 정책 확인"""
        
        # 전체 실행시간 확인
        start_time = context.get('start_time', time.time())
        elapsed = time.time() - start_time
        
        # 글로벌 타임아웃
        if global_policies and 'sla' in global_policies:
            sla_config = global_policies['sla']
            max_latency = sla_config.get('max_latency_seconds', 60)
            
            if elapsed > max_latency:
                logger.error(f"SLA violation: {elapsed:.2f}s > {max_latency}s")
                
                # Escalation 필요 여부
                if sla_config.get('escalate_on_violation', False):
                    return PolicyDecision(
                        action=PolicyAction.ESCALATE,
                        reason=f"SLA timeout ({elapsed:.2f}s)",
                        metadata={
                            'elapsed_time': elapsed,
                            'max_latency': max_latency,
                            'escalation_target': 'supervisor'
                        },
                        priority=95
                    )
                
                return PolicyDecision(
                    action=PolicyAction.TERMINATE,
                    reason=f"SLA timeout ({elapsed:.2f}s)",
                    metadata={'elapsed_time': elapsed},
                    priority=90
                )
        
        # 단계별 타임아웃
        if hasattr(step, 'timeout_seconds'):
            step_start = context.get(f'step_{step.name}_start', time.time())
            step_elapsed = time.time() - step_start
            
            if step_elapsed > step.timeout_seconds:
                logger.warning(f"Step timeout: {step.name} ({step_elapsed:.2f}s)")
                
                return PolicyDecision(
                    action=PolicyAction.SKIP,
                    reason=f"Step timeout ({step_elapsed:.2f}s)",
                    metadata={'step_name': step.name, 'elapsed': step_elapsed},
                    priority=75
                )
        
        return PolicyDecision(PolicyAction.CONTINUE, "SLA OK")
    
    def _check_confidence(self, step: Any, context: Dict[str, Any]) -> PolicyDecision:
        """신뢰도 임계값 확인"""
        
        if not hasattr(step, 'confidence_threshold'):
            return PolicyDecision(PolicyAction.CONTINUE, "No confidence check")
        
        threshold = step.confidence_threshold
        current_confidence = context.get('confidence', 1.0)
        
        if current_confidence < threshold:
            logger.info(f"Low confidence: {current_confidence:.2f} < {threshold}")
            
            # 낮은 신뢰도 처리 전략
            if current_confidence < threshold * 0.5:  # 매우 낮음
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason=f"Very low confidence ({current_confidence:.2f})",
                    metadata={
                        'confidence': current_confidence,
                        'threshold': threshold,
                        'escalation_target': 'expert'
                    },
                    priority=60
                )
            else:  # 약간 낮음
                return PolicyDecision(
                    action=PolicyAction.FALLBACK,
                    reason=f"Low confidence ({current_confidence:.2f})",
                    metadata={
                        'confidence': current_confidence,
                        'threshold': threshold,
                        'fallback_to': 'enhanced_rag'
                    },
                    priority=50
                )
        
        return PolicyDecision(PolicyAction.CONTINUE, "Confidence OK")
    
    async def _check_sentiment_policy(self, 
                                     step: Any,
                                     context: Dict[str, Any]) -> PolicyDecision:
        """감정 분석 기반 정책"""
        
        sentiment = context.get('sentiment', {})
        
        # 부정적 감정 처리
        if sentiment.get('label') == 'negative':
            score = sentiment.get('score', 0)
            
            if score > 0.8:  # 매우 부정적
                logger.warning(f"Very negative sentiment detected: {score:.2f}")
                
                return PolicyDecision(
                    action=PolicyAction.ESCALATE,
                    reason=f"Very negative sentiment ({score:.2f})",
                    metadata={
                        'sentiment': sentiment,
                        'escalation_target': 'senior_agent'
                    },
                    priority=70
                )
            
            elif score > 0.6:  # 부정적
                # 톤 조절 정책
                context['tone_adjustment'] = 'empathetic'
                
        # 긴급 키워드 감지
        urgent_keywords = ['긴급', '급해', '빨리', '당장', 'urgent', 'asap']
        user_message = context.get('user_message', '').lower()
        
        if any(keyword in user_message for keyword in urgent_keywords):
            return PolicyDecision(
                action=PolicyAction.ESCALATE,
                reason="Urgent request detected",
                metadata={'escalation_target': 'priority_queue'},
                priority=65
            )
        
        return PolicyDecision(PolicyAction.CONTINUE, "Sentiment OK")
    
    async def _evaluate_custom_policy(self, 
                                     policy: Dict[str, Any],
                                     context: Dict[str, Any]) -> PolicyDecision:
        """커스텀 정책 평가"""
        
        # 조건부 정책
        if 'conditions' in policy:
            conditions = [PolicyCondition(**c) for c in policy['conditions']]
            
            # AND/OR 로직 처리
            logic = policy.get('logic', 'AND')
            
            if logic == 'AND':
                condition_met = all(c.evaluate(context) for c in conditions)
            else:  # OR
                condition_met = any(c.evaluate(context) for c in conditions)
            
            if condition_met:
                action_str = policy.get('action', 'continue')
                
                try:
                    action = PolicyAction(action_str)
                    return PolicyDecision(
                        action=action,
                        reason=f"Custom policy triggered: {policy.get('name', 'unnamed')}",
                        metadata=policy.get('metadata', {}),
                        priority=policy.get('priority', 40)
                    )
                except ValueError:
                    logger.warning(f"Unknown policy action: {action_str}")
        
        # Retry 정책
        if 'retry' in policy:
            retry_config = policy['retry']
            max_retries = retry_config.get('max', 3)
            retry_count = context.get('retry_count', 0)
            
            if retry_count >= max_retries:
                return PolicyDecision(
                    action=PolicyAction.FALLBACK,
                    reason=f"Max retries ({max_retries}) exceeded",
                    metadata={'retry_count': retry_count},
                    priority=45
                )
            
            # 재시도 조건 확인
            if 'on_error' in retry_config:
                error_pattern = retry_config['on_error']
                last_error = context.get('last_error', '')
                
                if error_pattern in str(last_error):
                    return PolicyDecision(
                        action=PolicyAction.RETRY,
                        reason=f"Retrying on error: {error_pattern}",
                        metadata={
                            'retry_count': retry_count + 1,
                            'backoff': retry_config.get('backoff', 1.0)
                        },
                        priority=55
                    )
        
        # A/B 테스트 정책
        if 'ab_test' in policy:
            ab_config = policy['ab_test']
            variant = self._select_ab_variant(ab_config, context)
            
            context['ab_variant'] = variant
            logger.info(f"A/B test variant selected: {variant}")
            
            # 변형별 액션
            if 'variant_actions' in ab_config:
                variant_action = ab_config['variant_actions'].get(variant)
                if variant_action:
                    return PolicyDecision(
                        action=PolicyAction.FORK,
                        reason=f"A/B test variant: {variant}",
                        metadata={'variant': variant, 'action': variant_action},
                        priority=30
                    )
        
        # 캐싱 정책
        if 'cache' in policy:
            cache_config = policy['cache']
            cache_key = self._generate_cache_key(step, context, cache_config)
            
            if cache_config.get('check_cache', True):
                # 캐시 확인은 별도 구현 필요
                cached_result = context.get('cache', {}).get(cache_key)
                if cached_result:
                    return PolicyDecision(
                        action=PolicyAction.CACHE,
                        reason="Using cached result",
                        metadata={'cache_key': cache_key},
                        priority=20
                    )
        
        return PolicyDecision(PolicyAction.CONTINUE, "Custom policy OK")
    
    def _select_ab_variant(self, 
                          ab_config: Dict[str, Any],
                          context: Dict[str, Any]) -> str:
        """A/B 테스트 변형 선택"""
        
        # 세션 기반 일관된 변형 선택
        session_id = context.get('session_id', 'unknown')
        session_hash = hashlib.md5(session_id.encode()).hexdigest()
        hash_int = int(session_hash[:8], 16)
        
        # 가중치 기반 선택
        variants = ab_config.get('variants', [])
        if not variants:
            return 'control'
        
        total_weight = sum(v.get('weight', 1) for v in variants)
        threshold = hash_int % total_weight
        
        current = 0
        for variant in variants:
            current += variant.get('weight', 1)
            if threshold < current:
                return variant['name']
        
        return variants[0]['name']
    
    def _generate_cache_key(self, 
                           step: Any,
                           context: Dict[str, Any],
                           cache_config: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        
        # 키 구성 요소
        components = [
            step.name if hasattr(step, 'name') else 'unknown',
            context.get('user_message', ''),
        ]
        
        # 추가 키 필드
        if 'key_fields' in cache_config:
            for field in cache_config['key_fields']:
                value = context.get(field, '')
                components.append(str(value))
        
        # 해시 생성
        key_string = '|'.join(components)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _record_stats(self, step_name: str, decision: PolicyDecision):
        """정책 실행 통계 기록"""
        self.stats[step_name][decision.action.value] += 1
        self.stats['_global'][decision.action.value] += 1
        
        # 로그
        if decision.action != PolicyAction.CONTINUE:
            logger.info(
                f"Policy decision for {step_name}: {decision.action.value}",
                extra={'reason': decision.reason, 'metadata': decision.metadata}
            )
    
    def register_custom_handler(self, 
                               name: str, 
                               handler: Callable):
        """커스텀 정책 핸들러 등록"""
        self.custom_handlers[name] = handler
    
    async def apply_decision(self, 
                            decision: PolicyDecision,
                            step: Any,
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """정책 결정 적용"""
        
        result = {
            'action': decision.action.value,
            'reason': decision.reason,
            'metadata': decision.metadata
        }
        
        if decision.action == PolicyAction.RETRY:
            # 재시도 로직
            retry_count = decision.metadata.get('retry_count', 1)
            backoff = decision.metadata.get('backoff', 1.0)
            
            await asyncio.sleep(backoff * retry_count)
            context['retry_count'] = retry_count
            result['retry_after'] = backoff * retry_count
            
        elif decision.action == PolicyAction.FALLBACK:
            # 폴백 처리
            fallback_target = decision.metadata.get('fallback_model') or \
                            decision.metadata.get('fallback_to')
            
            if fallback_target:
                self.fallback_handler.record_fallback(
                    context.get('session_id', 'unknown'),
                    step.params.get('model', 'unknown') if hasattr(step, 'params') else 'unknown',
                    fallback_target,
                    decision.reason
                )
                result['fallback_to'] = fallback_target
        
        elif decision.action == PolicyAction.ESCALATE:
            # 에스컬레이션
            target = decision.metadata.get('escalation_target', 'human')
            escalation_result = await self.escalation_manager.escalate(
                context.get('session_id', 'unknown'),
                target,
                context
            )
            result['escalation'] = escalation_result
        
        elif decision.action == PolicyAction.THROTTLE:
            # 속도 제한
            wait_time = decision.metadata.get('wait_time', 1.0)
            await asyncio.sleep(wait_time)
            result['throttled_for'] = wait_time
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return {
            'by_step': dict(self.stats),
            'total_decisions': sum(
                sum(counts.values()) 
                for counts in self.stats.values()
            ),
            'top_actions': sorted(
                [(action, count) 
                 for action, count in self.stats['_global'].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    async def initialize_policies(self, config_file: Optional[str] = None):
        """정책 초기화"""
        
        if config_file:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # 에스컬레이션 규칙 등록
            if 'escalation_rules' in config:
                for rule_name, rule_config in config['escalation_rules'].items():
                    conditions = [
                        PolicyCondition(**c) 
                        for c in rule_config['conditions']
                    ]
                    self.escalation_manager.register_rule(
                        rule_name,
                        conditions,
                        rule_config['target'],
                        rule_config.get('priority', 0)
                    )
            
            # 폴백 체인 등록
            if 'fallback_chains' in config:
                for primary, fallbacks in config['fallback_chains'].items():
                    self.fallback_handler.register_fallback(primary, fallbacks)
            
            logger.info(f"Initialized policies from {config_file}")


# ========== Policy Configuration Loader ==========
class PolicyConfigLoader:
    """정책 설정 로더"""
    
    @staticmethod
    def load_from_yaml(file_path: str) -> Dict[str, Any]:
        """YAML 파일에서 정책 로드"""
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_from_json(file_path: str) -> Dict[str, Any]:
        """JSON 파일에서 정책 로드"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """정책 설정 검증"""
        required_keys = ['version', 'policies']
        
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required key in policy config: {key}")
                return False
        
        # 버전 확인
        if config['version'] != '1.0':
            logger.warning(f"Unsupported policy version: {config['version']}")
        
        return True


# ========== 사용 예시 ==========
if __name__ == "__main__":
    # 정책 엔진 초기화
    policy_engine = PolicyEngine({
        'cost_config': {
            'solar-10.7b': 0.001,
            'gpt-4': 0.03,
            'slm-ko-3b': 0.0005
        }
    })
    
    # 에스컬레이션 규칙 등록
    policy_engine.escalation_manager.register_rule(
        'angry_customer',
        [
            PolicyCondition('sentiment.label', '==', 'negative'),
            PolicyCondition('sentiment.score', '>', 0.8)
        ],
        'senior_agent',
        priority=90
    )
    
    # 폴백 체인 등록  
    policy_engine.fallback_handler.register_fallback(
        'gpt-4',
        [
            {'target': 'solar-10.7b', 'condition': {'field': 'cost_limit_reached', 'operator': '==', 'value': True}},
            {'target': 'slm-ko-3b'}
        ]
    )
    
    print("Policy Engine initialized successfully!")