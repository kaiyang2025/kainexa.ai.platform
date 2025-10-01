"""
tests/e2e/test_customer_service.py
고객 서비스 End-to-End 테스트
"""

import pytest
import asyncio
from datetime import datetime
import json
from httpx import AsyncClient

from tests.fixtures.workflows import CUSTOMER_SERVICE_WORKFLOW
from tests.fixtures.test_data import CUSTOMER_CONVERSATIONS


class TestCustomerServiceE2E:
    """고객 서비스 E2E 테스트"""
    
    @pytest.fixture
    async def setup_workflow(self, async_client, auth_headers):
        """고객 서비스 워크플로우 설정"""
        # 워크플로우 업로드
        response = await async_client.post(
            "/api/v1/workflows/upload",
            json={"dsl_content": json.dumps(CUSTOMER_SERVICE_WORKFLOW)},
            headers=auth_headers
        )
        workflow_id = response.json()["workflow_id"]
        
        # 컴파일
        await async_client.post(
            f"/api/v1/workflows/{workflow_id}/compile",
            json={"version": "1.0.0"},
            headers=auth_headers
        )
        
        # 퍼블리시
        await async_client.post(
            f"/api/v1/workflows/{workflow_id}/publish",
            json={"environment": "production"},
            headers=auth_headers
        )
        
        return workflow_id
    
    @pytest.mark.asyncio
    async def test_order_inquiry_flow(self, async_client, auth_headers, setup_workflow):
        """주문 조회 플로우 테스트"""
        # 세션 시작
        session = await self._start_session(async_client, auth_headers)
        
        # 1. 주문 조회 의도 전달
        response = await async_client.post(
            "/api/v1/workflow/customer_service/order_inquiry/execute",
            json={
                "input": {
                    "message": "제 주문 상태를 확인하고 싶어요",
                    "user_id": "customer_123"
                },
                "context": {
                    "session_id": session["session_id"]
                }
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "completed"
        assert "주문 번호" in result["response"]
        
        # 2. 주문 번호 제공
        response = await async_client.post(
            "/api/v1/workflow/customer_service/order_inquiry/execute",
            json={
                "input": {
                    "message": "ORD-2024-001234",
                    "user_id": "customer_123"
                },
                "context": {
                    "session_id": session["session_id"]
                }
            },
            headers=auth_headers
        )
        
        result = response.json()
        assert "배송" in result["response"]
        assert result["metadata"]["intent"] == "order_status"
    
    @pytest.mark.asyncio
    async def test_refund_request_flow(self, async_client, auth_headers, setup_workflow):
        """환불 요청 플로우 테스트"""
        session = await self._start_session(async_client, auth_headers)
        
        # 1. 환불 요청 시작
        response = await async_client.post(
            "/api/v1/workflow/customer_service/refund/execute",
            json={
                "input": {
                    "message": "환불하고 싶습니다",
                    "user_id": "customer_123"
                },
                "context": {
                    "session_id": session["session_id"]
                }
            },
            headers=auth_headers
        )
        
        result = response.json()
        assert "환불" in result["response"]
        assert "주문 번호" in result["response"]
        
        # 2. 주문 정보 제공
        response = await async_client.post(
            "/api/v1/workflow/customer_service/refund/execute",
            json={
                "input": {
                    "message": "주문번호는 ORD-2024-001234입니다",
                    "user_id": "customer_123"
                },
                "context": {
                    "session_id": session["session_id"]
                }
            },
            headers=auth_headers
        )
        
        result = response.json()
        assert "환불 사유" in result["response"]
        
        # 3. 환불 사유 제공
        response = await async_client.post(
            "/api/v1/workflow/customer_service/refund/execute",
            json={
                "input": {
                    "message": "제품에 하자가 있어요",
                    "user_id": "customer_123"
                },
                "context": {
                    "session_id": session["session_id"]
                }
            },
            headers=auth_headers
        )
        
        result = response.json()
        assert result["status"] == "completed"
        assert "환불 요청이 접수" in result["response"]
        assert result["metadata"]["action"] == "refund_initiated"
    
    @pytest.mark.asyncio
    async def test_escalation_to_human(self, async_client, auth_headers, setup_workflow):
        """상담사 에스컬레이션 테스트"""
        session = await self._start_session(async_client, auth_headers)
        
        # 복잡한 문의로 에스컬레이션 트리거
        response = await async_client.post(
            "/api/v1/workflow/customer_service/complex/execute",
            json={
                "input": {
                    "message": "제품 결함으로 인한 피해 보상과 법적 절차에 대해 문의합니다",
                    "user_id": "customer_123"
                },
                "context": {
                    "session_id": session["session_id"]
                }
            },
            headers=auth_headers
        )
        
        result = response.json()
        assert result["metadata"]["escalated"] == True
        assert result["metadata"]["escalation_reason"] == "complex_legal_inquiry"
        assert "상담사" in result["response"]
        assert "연결" in result["response"]
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, async_client, auth_headers, setup_workflow):
        """다중 턴 대화 테스트"""
        session = await self._start_session(async_client, auth_headers)
        
        conversation = [
            {
                "message": "안녕하세요",
                "expected": ["안녕", "무엇을 도와"]
            },
            {
                "message": "주문 관련 문의입니다",
                "expected": ["주문", "어떤"]
            },
            {
                "message": "배송이 늦어지고 있어요",
                "expected": ["주문 번호", "확인"]
            },
            {
                "message": "ORD-2024-005678",
                "expected": ["확인", "배송"]
            }
        ]
        
        for turn in conversation:
            response = await async_client.post(
                "/api/v1/workflow/customer_service/chat/execute",
                json={
                    "input": {
                        "message": turn["message"],
                        "user_id": "customer_123"
                    },
                    "context": {
                        "session_id": session["session_id"]
                    }
                },
                headers=auth_headers
            )
            
            result = response.json()
            assert result["status"] == "completed"
            
            # 예상 키워드 확인
            for keyword in turn["expected"]:
                assert keyword in result["response"]
            
            # 컨텍스트 유지 확인
            assert result["context"]["turn_count"] > 0
    
    @pytest.mark.asyncio
    async def test_product_recommendation(self, async_client, auth_headers, setup_workflow):
        """제품 추천 플로우 테스트"""
        session = await self._start_session(async_client, auth_headers)
        
        response = await async_client.post(
            "/api/v1/workflow/customer_service/recommend/execute",
            json={
                "input": {
                    "message": "노트북 추천해주세요",
                    "user_id": "customer_123"
                },
                "context": {
                    "session_id": session["session_id"],
                    "user_profile": {
                        "preferences": ["gaming", "portable"],
                        "budget": 1500000
                    }
                }
            },
            headers=auth_headers
        )
        
        result = response.json()
        assert result["status"] == "completed"
        assert "추천" in result["response"]
        assert len(result["metadata"]["recommendations"]) > 0
        assert all("price" in item for item in result["metadata"]["recommendations"])
    
    @pytest.mark.asyncio
    async def test_faq_handling(self, async_client, auth_headers, setup_workflow):
        """FAQ 처리 테스트"""
        faq_questions = [
            "영업시간이 어떻게 되나요?",
            "배송료는 얼마인가요?",
            "교환 정책이 궁금합니다",
            "회원 가입 혜택이 뭔가요?"
        ]
        
        for question in faq_questions:
            response = await async_client.post(
                "/api/v1/workflow/customer_service/faq/execute",
                json={
                    "input": {
                        "message": question,
                        "user_id": "customer_123"
                    }
                },
                headers=auth_headers
            )
            
            result = response.json()
            assert result["status"] == "completed"
            assert len(result["response"]) > 0
            assert result["metadata"]["source"] == "faq_database"
    
    @pytest.mark.asyncio
    async def test_sentiment_detection(self, async_client, auth_headers, setup_workflow):
        """감정 감지 및 대응 테스트"""
        test_cases = [
            {
                "message": "정말 화가 나네요! 이게 서비스입니까?",
                "expected_sentiment": "negative",
                "expected_response": ["죄송", "이해"]
            },
            {
                "message": "정말 만족스러운 서비스였습니다. 감사합니다!",
                "expected_sentiment": "positive",
                "expected_response": ["감사", "기쁘"]
            },
            {
                "message": "배송 상태 확인 부탁드립니다",
                "expected_sentiment": "neutral",
                "expected_response": ["확인", "도와"]
            }
        ]
        
        for test_case in test_cases:
            response = await async_client.post(
                "/api/v1/workflow/customer_service/chat/execute",
                json={
                    "input": {
                        "message": test_case["message"],
                        "user_id": "customer_123"
                    }
                },
                headers=auth_headers
            )
            
            result = response.json()
            assert result["metadata"]["sentiment"] == test_case["expected_sentiment"]
            
            # 감정에 맞는 응답 톤 확인
            for keyword in test_case["expected_response"]:
                assert keyword in result["response"]
    
    @pytest.mark.asyncio
    async def test_language_detection(self, async_client, auth_headers, setup_workflow):
        """다국어 감지 및 처리 테스트"""
        multilingual_queries = [
            {"message": "I need help with my order", "lang": "en"},
            {"message": "私の注文について助けが必要です", "lang": "ja"},
            {"message": "我需要订单帮助", "lang": "zh"},
            {"message": "주문 관련 도움이 필요합니다", "lang": "ko"}
        ]
        
        for query in multilingual_queries:
            response = await async_client.post(
                "/api/v1/workflow/customer_service/multilingual/execute",
                json={
                    "input": {
                        "message": query["message"],
                        "user_id": "customer_123"
                    }
                },
                headers=auth_headers
            )
            
            result = response.json()
            assert result["status"] == "completed"
            assert result["metadata"]["detected_language"] == query["lang"]
            # 응답이 같은 언어로 되어있는지 확인
            assert result["metadata"]["response_language"] == query["lang"]
    
    async def _start_session(self, async_client, auth_headers):
        """세션 시작 헬퍼"""
        response = await async_client.post(
            "/api/v1/sessions/start",
            json={
                "user_id": "customer_123",
                "channel": "web"
            },
            headers=auth_headers
        )
        return response.json()