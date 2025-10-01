"""
tests/unit/test_model_router.py
모델 라우터 단위 테스트
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from src.core.orchestration.model_router import (
    ModelRouter,
    ModelType,
    ModelProfile,
    RoutingStrategy,
    LoadBalancer,
    ModelHealth
)


class TestModelRouter:
    """모델 라우터 테스트"""
    
    @pytest.fixture
    def model_router(self):
        """모델 라우터 인스턴스"""
        config = {
            "models": [
                {
                    "name": "gpt-4",
                    "type": "large",
                    "endpoint": "http://gpt4:8000",
                    "max_tokens": 4000,
                    "cost_per_token": 0.00003,
                    "priority": 1
                },
                {
                    "name": "gpt-3.5-turbo", 
                    "type": "medium",
                    "endpoint": "http://gpt35:8000",
                    "max_tokens": 2000,
                    "cost_per_token": 0.000015,
                    "priority": 2
                },
                {
                    "name": "llama-2-7b",
                    "type": "small",
                    "endpoint": "http://llama:8000",
                    "max_tokens": 1000,
                    "cost_per_token": 0.000005,
                    "priority": 3
                }
            ],
            "routing_strategy": "adaptive",
            "fallback_enabled": True
        }
        return ModelRouter(config)
    
    @pytest.mark.asyncio
    async def test_route_by_complexity(self, model_router):
        """복잡도 기반 라우팅 테스트"""
        # 복잡한 쿼리
        complex_query = {
            "prompt": "Explain quantum computing with detailed mathematical formulations",
            "complexity_score": 0.9,
            "expected_tokens": 2000
        }
        
        selected_model = await model_router.route(complex_query)
        assert selected_model.name == "gpt-4"
        
        # 간단한 쿼리
        simple_query = {
            "prompt": "What is the capital of France?",
            "complexity_score": 0.2,
            "expected_tokens": 50
        }
        
        selected_model = await model_router.route(simple_query)
        assert selected_model.name in ["llama-2-7b", "gpt-3.5-turbo"]
    
    @pytest.mark.asyncio
    async def test_cost_optimization(self, model_router):
        """비용 최적화 라우팅 테스트"""
        model_router.routing_strategy = RoutingStrategy.COST_OPTIMIZED
        
        query = {
            "prompt": "Standard query",
            "max_budget": 0.01,
            "expected_tokens": 500
        }
        
        selected_model = await model_router.route(query)
        
        # 가장 저렴한 모델 선택
        expected_cost = selected_model.cost_per_token * 500
        assert expected_cost <= 0.01
        assert selected_model.name == "llama-2-7b"
    
    @pytest.mark.asyncio
    async def test_latency_optimization(self, model_router):
        """레이턴시 최적화 라우팅 테스트"""
        model_router.routing_strategy = RoutingStrategy.LATENCY_OPTIMIZED
        
        # 모델 응답 시간 시뮬레이션
        with patch.object(model_router, 'get_model_latency') as mock_latency:
            mock_latency.side_effect = lambda m: {
                "gpt-4": 2000,
                "gpt-3.5-turbo": 800,
                "llama-2-7b": 300
            }.get(m.name, 1000)
            
            query = {"prompt": "Quick response needed", "sla_ms": 500}
            selected_model = await model_router.route(query)
            
            assert selected_model.name == "llama-2-7b"
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, model_router):
        """폴백 메커니즘 테스트"""
        query = {"prompt": "Test query"}
        
        # 첫 번째 모델 실패 시뮬레이션
        with patch.object(model_router, 'check_model_health') as mock_health:
            mock_health.side_effect = [
                ModelHealth.UNHEALTHY,  # gpt-4 불가
                ModelHealth.HEALTHY,    # gpt-3.5-turbo 가능
            ]
            
            selected_model = await model_router.route_with_fallback(query)
            assert selected_model.name == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, model_router):
        """로드 밸런싱 테스트"""
        load_balancer = LoadBalancer(strategy="round_robin")
        
        models = [
            ModelProfile("model1", ModelType.MEDIUM),
            ModelProfile("model2", ModelType.MEDIUM),
            ModelProfile("model3", ModelType.MEDIUM)
        ]
        
        # 라운드 로빈 분배
        selections = []
        for _ in range(6):
            selected = await load_balancer.select(models)
            selections.append(selected.name)
        
        # 균등 분배 확인
        assert selections.count("model1") == 2
        assert selections.count("model2") == 2
        assert selections.count("model3") == 2
    
    @pytest.mark.asyncio
    async def test_adaptive_routing(self, model_router):
        """적응형 라우팅 테스트"""
        model_router.routing_strategy = RoutingStrategy.ADAPTIVE
        
        # 과거 성능 데이터 설정
        with patch.object(model_router, 'get_historical_performance') as mock_perf:
            mock_perf.return_value = {
                "gpt-4": {"success_rate": 0.95, "avg_latency": 1500},
                "gpt-3.5-turbo": {"success_rate": 0.92, "avg_latency": 800},
                "llama-2-7b": {"success_rate": 0.88, "avg_latency": 400}
            }
            
            # 고품질 요구사항
            high_quality_query = {
                "prompt": "Critical analysis needed",
                "quality_requirement": "high"
            }
            selected = await model_router.route(high_quality_query)
            assert selected.name == "gpt-4"
            
            # 속도 우선
            speed_query = {
                "prompt": "Quick classification",
                "quality_requirement": "acceptable",
                "max_latency": 500
            }
            selected = await model_router.route(speed_query)
            assert selected.name == "llama-2-7b"
    
    @pytest.mark.asyncio
    async def test_token_limit_routing(self, model_router):
        """토큰 제한 기반 라우팅 테스트"""
        # 긴 컨텍스트 요청
        long_context_query = {
            "prompt": "Long text...",
            "context_tokens": 3500,
            "expected_output_tokens": 500
        }
        
        selected_model = await model_router.route(long_context_query)
        
        # GPT-4만 4000 토큰 처리 가능
        assert selected_model.name == "gpt-4"
        assert selected_model.max_tokens >= 4000
    
    @pytest.mark.asyncio
    async def test_model_health_monitoring(self, model_router):
        """모델 헬스 모니터링 테스트"""
        health_monitor = model_router.health_monitor
        
        # 헬스 체크
        model = ModelProfile("test-model", ModelType.LARGE)
        model.endpoint = "http://test:8000"
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            # 성공 응답
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            mock_get.return_value.__aenter__.return_value = mock_response
            
            health = await health_monitor.check_health(model)
            assert health == ModelHealth.HEALTHY
            
            # 실패 응답
            mock_response.status = 500
            health = await health_monitor.check_health(model)
            assert health == ModelHealth.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, model_router):
        """재시도 및 백오프 테스트"""
        query = {"prompt": "Test retry"}
        
        attempt_count = 0
        
        async def mock_inference(model, query):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return {"response": "Success"}
        
        with patch.object(model_router, 'execute_inference', mock_inference):
            result = await model_router.route_with_retry(
                query,
                max_retries=3,
                backoff_factor=0.1
            )
            
            assert attempt_count == 3
            assert result["response"] == "Success"
    
    @pytest.mark.asyncio
    async def test_ab_testing_routing(self, model_router):
        """A/B 테스팅 라우팅 테스트"""
        ab_config = {
            "experiment_id": "exp_001",
            "variant_a": {"model": "gpt-4", "traffic": 0.3},
            "variant_b": {"model": "gpt-3.5-turbo", "traffic": 0.7}
        }
        
        model_router.enable_ab_testing(ab_config)
        
        # 트래픽 분배 테스트
        selections = {"gpt-4": 0, "gpt-3.5-turbo": 0}
        
        for i in range(1000):
            with patch('random.random', return_value=i/1000):
                selected = await model_router.route({"prompt": "test"})
                selections[selected.name] += 1
        
        # 대략적인 트래픽 분배 확인 (오차 5% 허용)
        assert abs(selections["gpt-4"] / 1000 - 0.3) < 0.05
        assert abs(selections["gpt-3.5-turbo"] / 1000 - 0.7) < 0.05


class TestModelProfiles:
    """모델 프로파일 테스트"""
    
    def test_model_profile_creation(self):
        """모델 프로파일 생성 테스트"""
        profile = ModelProfile(
            name="custom-model",
            type=ModelType.LARGE,
            endpoint="http://custom:8000",
            max_tokens=4096,
            cost_per_token=0.00002,
            supported_languages=["en", "ko", "ja"],
            capabilities=["chat", "completion", "embedding"]
        )
        
        assert profile.name == "custom-model"
        assert profile.type == ModelType.LARGE
        assert "ko" in profile.supported_languages
        assert "embedding" in profile.capabilities
    
    def test_model_profile_comparison(self):
        """모델 프로파일 비교 테스트"""
        profile1 = ModelProfile("model1", ModelType.LARGE, cost_per_token=0.00003)
        profile2 = ModelProfile("model2", ModelType.MEDIUM, cost_per_token=0.00001)
        
        # 비용 비교
        assert profile1.is_more_expensive_than(profile2)
        
        # 크기 비교
        assert profile1.is_larger_than(profile2)
    
    def test_model_capability_matching(self):
        """모델 능력 매칭 테스트"""
        profile = ModelProfile(
            name="multi-modal",
            type=ModelType.LARGE,
            capabilities=["text", "image", "code"]
        )
        
        # 능력 확인
        assert profile.supports("text")
        assert profile.supports("image")
        assert not profile.supports("audio")
        
        # 다중 능력 확인
        assert profile.supports_all(["text", "code"])
        assert not profile.supports_all(["text", "audio"])


class TestRoutingStrategies:
    """라우팅 전략 테스트"""
    
    @pytest.mark.asyncio
    async def test_weighted_random_strategy(self):
        """가중 랜덤 전략 테스트"""
        from src.core.orchestration.model_router import WeightedRandomStrategy
        
        strategy = WeightedRandomStrategy()
        
        models = [
            ModelProfile("high", ModelType.LARGE, weight=0.5),
            ModelProfile("medium", ModelType.MEDIUM, weight=0.3),
            ModelProfile("low", ModelType.SMALL, weight=0.2)
        ]
        
        selections = {"high": 0, "medium": 0, "low": 0}
        
        for _ in range(1000):
            selected = await strategy.select(models, {})
            selections[selected.name] += 1
        
        # 가중치에 따른 분배 확인 (오차 5% 허용)
        assert abs(selections["high"] / 1000 - 0.5) < 0.05
        assert abs(selections["medium"] / 1000 - 0.3) < 0.05
        assert abs(selections["low"] / 1000 - 0.2) < 0.05
    
    @pytest.mark.asyncio
    async def test_least_loaded_strategy(self):
        """최소 부하 전략 테스트"""
        from src.core.orchestration.model_router import LeastLoadedStrategy
        
        strategy = LeastLoadedStrategy()
        
        models = [
            ModelProfile("model1", ModelType.MEDIUM),
            ModelProfile("model2", ModelType.MEDIUM),
            ModelProfile("model3", ModelType.MEDIUM)
        ]
        
        # 부하 상태 설정
        with patch.object(strategy, 'get_current_load') as mock_load:
            mock_load.side_effect = lambda m: {
                "model1": 0.8,  # 80% 부하
                "model2": 0.3,  # 30% 부하
                "model3": 0.5   # 50% 부하
            }.get(m.name, 0)
            
            selected = await strategy.select(models, {})
            assert selected.name == "model2"  # 가장 낮은 부하
    
    @pytest.mark.asyncio
    async def test_priority_based_strategy(self):
        """우선순위 기반 전략 테스트"""
        from src.core.orchestration.model_router import PriorityBasedStrategy
        
        strategy = PriorityBasedStrategy()
        
        models = [
            ModelProfile("low-priority", ModelType.SMALL, priority=3),
            ModelProfile("high-priority", ModelType.LARGE, priority=1),
            ModelProfile("medium-priority", ModelType.MEDIUM, priority=2)
        ]
        
        # 모든 모델 건강한 상태
        with patch.object(strategy, 'is_healthy', return_value=True):
            selected = await strategy.select(models, {})
            assert selected.name == "high-priority"
        
        # 높은 우선순위 모델 불가
        with patch.object(strategy, 'is_healthy') as mock_health:
            mock_health.side_effect = lambda m: m.name != "high-priority"
            selected = await strategy.select(models, {})
            assert selected.name == "medium-priority"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])