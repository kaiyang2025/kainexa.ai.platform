# tests/unit/test_policy_engine.py
"""
Policy Engine 테스트 스위트
정책 평가, 에스컬레이션, 폴백, 비용 관리 등 테스트
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.orchestration.policy_engine import (
    PolicyEngine, 
    PolicyAction, 
    PolicyDecision,
    PolicyCondition,
    RateLimiter,
    CostTracker,
    FallbackHandler,
    EscalationManager,
    PolicyConfigLoader
)

# ========== Fixtures ==========
@pytest.fixture
def policy_engine():
    """Policy Engine 인스턴스"""
    config = {
        'cost_config': {
            'solar-10.7b': 0.001,
            'gpt-4': 0.03,
            'slm-ko-3b': 0.0005
        }
    }
    return PolicyEngine(config)

@pytest.fixture
def sample_step():
    """샘플 스텝 객체"""
    step = Mock()
    step.name = "test_step"
    step.type = "llm"
    step.params = {
        'model': 'solar-10.7b',
        'max_tokens': 512
    }
    return step

@pytest.fixture
def sample_context():
    """샘플 실행 컨텍스트"""
    return {
        'session_id': 'test-session-123',
        'tenant_id': 'tenant-001',
        'start_time': time.time(),
        'user_message': 'test message',
        'confidence': 0.8,
        'sentiment': {'label': 'neutral', 'score': 0.5},
        'language': 'ko-KR'
    }

@pytest.fixture
def global_policies():
    """샘플 글로벌 정책"""
    return {
        'sla': {
            'max_latency_seconds': 30,
            'escalate_on_violation': True
        },
        'cost_limit': {
            'max_per_session': 1.0,
            'fallback_model': 'slm-ko-3b',
            'monthly_budget': 10000
        }
    }

# ========== Policy Condition Tests ==========
class TestPolicyCondition:
    """PolicyCondition 테스트"""
    
    def test_equality_operator(self):
        """동등 연산자 테스트"""
        condition = PolicyCondition('status', '==', 'active')
        assert condition.evaluate({'status': 'active'}) == True
        assert condition.evaluate({'status': 'inactive'}) == False
    
    def test_comparison_operators(self):
        """비교 연산자 테스트"""
        # Greater than
        condition = PolicyCondition('score', '>', 0.5)
        assert condition.evaluate({'score': 0.6}) == True
        assert condition.evaluate({'score': 0.4}) == False
        
        # Less than or equal
        condition = PolicyCondition('count', '<=', 10)
        assert condition.evaluate({'count': 10}) == True
        assert condition.evaluate({'count': 11}) == False
    
    def test_membership_operators(self):
        """멤버십 연산자 테스트"""
        # In operator
        condition = PolicyCondition('status', 'in', ['active', 'pending'])
        assert condition.evaluate({'status': 'active'}) == True
        assert condition.evaluate({'status': 'inactive'}) == False
        
        # Not in operator
        condition = PolicyCondition('role', 'not_in', ['admin', 'root'])
        assert condition.evaluate({'role': 'user'}) == True
        assert condition.evaluate({'role': 'admin'}) == False
    
    def test_regex_operator(self):
        """정규식 연산자 테스트"""
        condition = PolicyCondition('message', 'regex', r'.*urgent.*')
        assert condition.evaluate({'message': 'this is urgent!'}) == True
        assert condition.evaluate({'message': 'normal message'}) == False
    
    def test_exists_operator(self):
        """존재 연산자 테스트"""
        condition = PolicyCondition('optional_field', 'exists', True)
        assert condition.evaluate({'optional_field': 'value'}) == True
        assert condition.evaluate({'other_field': 'value'}) == False
    
    def test_nested_field_access(self):
        """중첩 필드 접근 테스트"""
        condition = PolicyCondition('user.profile.age', '>', 18)
        
        context = {
            'user': {
                'profile': {
                    'age': 25
                }
            }
        }
        assert condition.evaluate(context) == True
        
        context['user']['profile']['age'] = 15
        assert condition.evaluate(context) == False

# ========== Rate Limiter Tests ==========
class TestRateLimiter:
    """RateLimiter 테스트"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Rate limiting 기본 동작"""
        limiter = RateLimiter()
        key = "test_key"
        
        # 처음 3개 요청은 허용
        for _ in range(3):
            allowed, wait = await limiter.check_limit(key, 3, 60)
            assert allowed == True
            assert wait == 0
        
        # 4번째 요청은 차단
        allowed, wait = await limiter.check_limit(key, 3, 60)
        assert allowed == False
        assert wait > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_expiry(self):
        """Rate limit 만료 테스트"""
        limiter = RateLimiter()
        key = "test_key"
        
        # 3개 요청 사용
        for _ in range(3):
            await limiter.check_limit(key, 3, 1)  # 1초 window
        
        # 즉시 4번째는 차단
        allowed, wait = await limiter.check_limit(key, 3, 1)
        assert allowed == False
        
        # 1초 후에는 허용
        await asyncio.sleep(1.1)
        allowed, wait = await limiter.check_limit(key, 3, 1)
        assert allowed == True
    
    def test_rate_limit_reset(self):
        """Rate limit 리셋 테스트"""
        limiter = RateLimiter()
        key = "test_key"
        
        # 요청 기록 추가
        limiter.requests[key] = [time.time()]
        assert len(limiter.requests[key]) == 1
        
        # 리셋
        limiter.reset(key)
        assert key not in limiter.requests

# ========== Cost Tracker Tests ==========
class TestCostTracker:
    """CostTracker 테스트"""
    
    def test_session_cost_tracking(self):
        """세션 비용 추적"""
        tracker = CostTracker()
        
        # 비용 추가
        tracker.add_cost('session1', 0.5)
        tracker.add_cost('session1', 0.3)
        tracker.add_cost('session2', 0.2)
        
        assert tracker.get_session_cost('session1') == 0.8
        assert tracker.get_session_cost('session2') == 0.2
        assert tracker.get_session_cost('session3') == 0.0
    
    def test_llm_cost_estimation(self):
        """LLM 비용 예측"""
        tracker = CostTracker({
            'gpt-4': 0.03,
            'solar-10.7b': 0.001
        })
        
        # Input 비용
        cost = tracker.estimate_llm_cost('gpt-4', 1000, is_input=True)
        assert cost == 0.03
        
        # Output 비용 (2배)
        cost = tracker.estimate_llm_cost('gpt-4', 1000, is_input=False)
        assert cost == 0.06
        
        # 다른 모델
        cost = tracker.estimate_llm_cost('solar-10.7b', 5000, is_input=True)
        assert cost == 0.005
    
    def test_budget_checking(self):
        """예산 확인"""
        tracker = CostTracker()
        tenant_id = 'tenant1'
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 일일 비용 추가
        tracker.daily_costs[tenant_id][today] = 450
        
        # 예산 확인
        within, remaining = tracker.check_budget(tenant_id, 'daily', 500)
        assert within == True
        assert remaining == 50
        
        # 예산 초과
        tracker.daily_costs[tenant_id][today] = 550
        within, remaining = tracker.check_budget(tenant_id, 'daily', 500)
        assert within == False
        assert remaining == -50

# ========== Fallback Handler Tests ==========
class TestFallbackHandler:
    """FallbackHandler 테스트"""
    
    def test_fallback_registration(self):
        """폴백 체인 등록"""
        handler = FallbackHandler()
        
        handler.register_fallback('gpt-4', [
            {'target': 'gpt-3.5', 'condition': {'field': 'error', 'operator': '==', 'value': 'rate_limit'}},
            {'target': 'solar-10.7b'}
        ])
        
        assert 'gpt-4' in handler.fallback_chains
        assert len(handler.fallback_chains['gpt-4']) == 2
    
    def test_fallback_selection_with_condition(self):
        """조건부 폴백 선택"""
        handler = FallbackHandler()
        
        handler.register_fallback('primary', [
            {'target': 'secondary', 'condition': {'field': 'error', 'operator': '==', 'value': 'timeout'}},
            {'target': 'tertiary'}
        ])
        
        # 조건 매칭
        context = {'error': 'timeout'}
        fallback = handler.get_fallback('primary', context)
        assert fallback == 'secondary'
        
        # 조건 미매칭 - 다음 폴백
        context = {'error': 'other'}
        fallback = handler.get_fallback('primary', context)
        assert fallback == 'tertiary'
    
    def test_fallback_history_recording(self):
        """폴백 이력 기록"""
        handler = FallbackHandler()
        
        handler.record_fallback('session1', 'gpt-4', 'gpt-3.5', 'cost limit')
        handler.record_fallback('session1', 'gpt-3.5', 'solar', 'rate limit')
        
        assert len(handler.fallback_history['session1']) == 2
        assert handler.fallback_history['session1'][0]['from'] == 'gpt-4'
        assert handler.fallback_history['session1'][1]['to'] == 'solar'

# ========== Escalation Manager Tests ==========
class TestEscalationManager:
    """EscalationManager 테스트"""
    
    def test_escalation_rule_registration(self):
        """에스컬레이션 규칙 등록"""
        manager = EscalationManager()
        
        conditions = [
            PolicyCondition('sentiment.score', '>', 0.8),
            PolicyCondition('sentiment.label', '==', 'negative')
        ]
        
        manager.register_rule('angry_customer', conditions, 'senior_agent', priority=90)
        
        assert 'angry_customer' in manager.escalation_rules
        assert manager.escalation_rules['angry_customer']['priority'] == 90
    
    @pytest.mark.asyncio
    async def test_escalation_check(self):
        """에스컬레이션 필요 확인"""
        manager = EscalationManager()
        
        # 규칙 등록
        manager.register_rule(
            'high_value',
            [PolicyCondition('amount', '>', 10000)],
            'supervisor',
            priority=80
        )
        
        manager.register_rule(
            'vip_customer',
            [PolicyCondition('customer_tier', '==', 'VIP')],
            'vip_specialist',
            priority=95
        )
        
        # VIP 고객 - 더 높은 우선순위
        context = {'amount': 20000, 'customer_tier': 'VIP'}
        result = await manager.check_escalation(context)
        
        assert result is not None
        assert result['target'] == 'vip_specialist'
        assert result['priority'] == 95
    
    @pytest.mark.asyncio
    async def test_escalation_execution(self):
        """에스컬레이션 실행"""
        manager = EscalationManager()
        
        # 상담사 풀 추가
        manager.agent_pools['supervisor'] = ['agent1', 'agent2']
        
        # 에스컬레이션 실행
        result = await manager.escalate('session1', 'supervisor', {'reason': 'test'})
        
        assert result['session_id'] == 'session1'
        assert result['target'] == 'supervisor'
        assert result['assigned_to'] == 'agent1'
        assert result['status'] == 'assigned'
        
        # 상담사 풀이 줄어듦
        assert len(manager.agent_pools['supervisor']) == 1

# ========== Main Policy Engine Tests ==========
class TestPolicyEngine:
    """PolicyEngine 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_policy(self, policy_engine, sample_step, sample_context):
        """Rate limiting 정책 테스트"""
        # Rate limit 설정 추가
        sample_step.rate_limit = {
            'max_requests': 1,
            'period_seconds': 60,
            'scope': 'session'
        }
        
        # 첫 요청은 통과
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.CONTINUE
        
        # 두 번째 요청은 차단
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.THROTTLE
        assert 'wait_time' in decision.metadata
    
    @pytest.mark.asyncio
    async def test_cost_limit_policy(self, policy_engine, sample_step, sample_context, global_policies):
        """비용 제한 정책 테스트"""
        # 세션 비용을 한도에 도달시킴
        policy_engine.cost_tracker.add_cost(sample_context['session_id'], 1.0)
        
        decision = await policy_engine.evaluate(sample_step, sample_context, global_policies)
        assert decision.action == PolicyAction.FALLBACK
        assert decision.metadata['fallback_model'] == 'slm-ko-3b'
    
    @pytest.mark.asyncio
    async def test_sla_timeout_policy(self, policy_engine, sample_step, sample_context, global_policies):
        """SLA 타임아웃 정책 테스트"""
        # 시작 시간을 과거로 설정 (타임아웃 유발)
        sample_context['start_time'] = time.time() - 35  # 35초 전
        
        decision = await policy_engine.evaluate(sample_step, sample_context, global_policies)
        assert decision.action == PolicyAction.ESCALATE
        assert 'elapsed_time' in decision.metadata
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_policy(self, policy_engine, sample_step, sample_context):
        """신뢰도 임계값 정책 테스트"""
        sample_step.confidence_threshold = 0.7
        
        # 높은 신뢰도
        sample_context['confidence'] = 0.8
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.CONTINUE
        
        # 낮은 신뢰도
        sample_context['confidence'] = 0.3
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.ESCALATE
        assert decision.metadata['escalation_target'] == 'expert'
    
    @pytest.mark.asyncio
    async def test_sentiment_policy(self, policy_engine, sample_step, sample_context):
        """감정 분석 정책 테스트"""
        # 매우 부정적 감정
        sample_context['sentiment'] = {
            'label': 'negative',
            'score': 0.9
        }
        
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.ESCALATE
        assert decision.metadata['escalation_target'] == 'senior_agent'
    
    @pytest.mark.asyncio
    async def test_urgent_keyword_detection(self, policy_engine, sample_step, sample_context):
        """긴급 키워드 감지 테스트"""
        sample_context['user_message'] = '긴급! 빨리 처리해주세요!'
        
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.ESCALATE
        assert decision.metadata['escalation_target'] == 'priority_queue'
    
    @pytest.mark.asyncio
    async def test_custom_policy_conditions(self, policy_engine, sample_step, sample_context):
        """커스텀 정책 조건 테스트"""
        sample_step.policy = {
            'conditions': [
                {'field': 'language', 'operator': '==', 'value': 'ko-KR'},
                {'field': 'confidence', 'operator': '>', 'value': 0.7}
            ],
            'logic': 'AND',
            'action': 'continue',
            'priority': 50
        }
        
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.CONTINUE
    
    @pytest.mark.asyncio
    async def test_retry_policy(self, policy_engine, sample_step, sample_context):
        """재시도 정책 테스트"""
        sample_step.policy = {
            'retry': {
                'max': 3,
                'on_error': 'timeout',
                'backoff': 2.0
            }
        }
        
        # 에러 발생 상황
        sample_context['last_error'] = 'Connection timeout'
        sample_context['retry_count'] = 1
        
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.RETRY
        assert decision.metadata['retry_count'] == 2
    
    @pytest.mark.asyncio
    async def test_ab_test_policy(self, policy_engine, sample_step, sample_context):
        """A/B 테스트 정책"""
        sample_step.policy = {
            'ab_test': {
                'variants': [
                    {'name': 'control', 'weight': 50},
                    {'name': 'experiment', 'weight': 50}
                ],
                'variant_actions': {
                    'control': 'use_standard_prompt',
                    'experiment': 'use_enhanced_prompt'
                }
            }
        }
        
        decision = await policy_engine.evaluate(sample_step, sample_context)
        
        # 세션 기반으로 일관된 변형 선택
        assert 'ab_variant' in sample_context
        assert sample_context['ab_variant'] in ['control', 'experiment']
    
    @pytest.mark.asyncio
    async def test_policy_priority_ordering(self, policy_engine, sample_step, sample_context):
        """정책 우선순위 순서 테스트"""
        # 여러 정책 위반 상황 설정
        sample_step.rate_limit = {
            'max_requests': 0,  # 즉시 차단
            'period_seconds': 60,
            'scope': 'session'
        }
        sample_context['confidence'] = 0.3  # 낮은 신뢰도
        
        # Rate limit이 더 높은 우선순위
        decision = await policy_engine.evaluate(sample_step, sample_context)
        assert decision.action == PolicyAction.THROTTLE  # Rate limit이 우선
    
    @pytest.mark.asyncio
    async def test_apply_decision(self, policy_engine, sample_step, sample_context):
        """정책 결정 적용 테스트"""
        # Retry 결정
        decision = PolicyDecision(
            PolicyAction.RETRY,
            "Test retry",
            {'retry_count': 2, 'backoff': 1.0}
        )
        
        result = await policy_engine.apply_decision(decision, sample_step, sample_context)
        assert result['action'] == 'retry'
        assert result['retry_after'] == 2.0
        assert sample_context['retry_count'] == 2
    
    def test_policy_stats_recording(self, policy_engine):
        """정책 통계 기록 테스트"""
        # 통계 기록
        policy_engine._record_stats('step1', PolicyDecision(PolicyAction.CONTINUE, "OK"))
        policy_engine._record_stats('step1', PolicyDecision(PolicyAction.RETRY, "Error"))
        policy_engine._record_stats('step2', PolicyDecision(PolicyAction.ESCALATE, "Low confidence"))
        
        stats = policy_engine.get_stats()
        
        assert stats['by_step']['step1']['continue'] == 1
        assert stats['by_step']['step1']['retry'] == 1
        assert stats['by_step']['step2']['escalate'] == 1
        assert stats['total_decisions'] == 3

# ========== Config Loader Tests ==========
class TestPolicyConfigLoader:
    """PolicyConfigLoader 테스트"""
    
    def test_validate_config(self):
        """설정 검증 테스트"""
        # 유효한 설정
        valid_config = {
            'version': '1.0',
            'policies': {}
        }
        assert PolicyConfigLoader.validate_config(valid_config) == True
        
        # 필수 키 누락
        invalid_config = {'version': '1.0'}
        assert PolicyConfigLoader.validate_config(invalid_config) == False
    
    @patch('builtins.open', create=True)
    @patch('yaml.safe_load')
    def test_load_from_yaml(self, mock_yaml_load, mock_open):
        """YAML 로드 테스트"""
        mock_yaml_load.return_value = {'version': '1.0', 'policies': {}}
        
        config = PolicyConfigLoader.load_from_yaml('test.yaml')
        assert config['version'] == '1.0'
        mock_open.assert_called_once_with('test.yaml', 'r')

# ========== Integration Tests ==========
class TestPolicyEngineIntegration:
    """Policy Engine 통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_policy_flow(self):
        """전체 정책 플로우 테스트"""
        # Policy Engine 초기화
        engine = PolicyEngine()
        
        # 에스컬레이션 규칙 등록
        engine.escalation_manager.register_rule(
            'test_rule',
            [PolicyCondition('test_flag', '==', True)],
            'supervisor',
            priority=80
        )
        
        # 폴백 체인 등록
        engine.fallback_handler.register_fallback(
            'primary_model',
            [{'target': 'fallback_model'}]
        )
        
        # 스텝과 컨텍스트 생성
        step = Mock()
        step.name = "integration_test"
        step.type = "llm"
        step.params = {'model': 'primary_model'}
        
        context = {
            'session_id': 'integration-test',
            'start_time': time.time(),
            'test_flag': True
        }
        
        # 정책 평가
        decision = await engine.evaluate(step, context)
        
        # 에스컬레이션이 트리거되어야 함
        assert decision.action == PolicyAction.ESCALATE
        assert decision.metadata['escalation_target'] == 'supervisor'
    
    @pytest.mark.asyncio
    async def test_policy_chain_execution(self):
        """정책 체인 실행 테스트"""
        engine = PolicyEngine()
        
        # 복잡한 정책 시나리오
        step = Mock()
        step.name = "chain_test"
        step.type = "llm"
        step.confidence_threshold = 0.9
        step.policy = {
            'retry': {'max': 2},
            'cache': {'check_cache': True}
        }
        
        context = {
            'session_id': 'chain-test',
            'start_time': time.time(),
            'confidence': 0.95,
            'retry_count': 0
        }
        
        # 첫 번째 평가 - 통과
        decision = await engine.evaluate(step, context)
        assert decision.action == PolicyAction.CONTINUE
        
        # 신뢰도 낮춤
        context['confidence'] = 0.5
        decision = await engine.evaluate(step, context)
        assert decision.action == PolicyAction.FALLBACK
        
        # 재시도 횟수 초과
        context['retry_count'] = 3
        decision = await engine.evaluate(step, context)
        assert decision.action in [PolicyAction.FALLBACK, PolicyAction.ESCALATE]


# ========== Performance Tests ==========
class TestPolicyEnginePerformance:
    """Policy Engine 성능 테스트"""
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, policy_engine):
        """동시 평가 성능 테스트"""
        step = Mock()
        step.name = "perf_test"
        step.rate_limit = {
            'max_requests': 100,
            'period_seconds': 1,
            'scope': 'global'
        }
        
        # 100개 동시 요청
        tasks = []
        for i in range(100):
            context = {
                'session_id': f'perf-{i}',
                'start_time': time.time()
            }
            tasks.append(policy_engine.evaluate(step, context))
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        
        # 1초 이내 완료
        assert duration < 1.0
        
        # 결과 검증
        continue_count = sum(1 for r in results if r.action == PolicyAction.CONTINUE)
        throttle_count = sum(1 for r in results if r.action == PolicyAction.THROTTLE)
        
        assert continue_count == 100  # Rate limit이 100이므로
        assert throttle_count == 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])