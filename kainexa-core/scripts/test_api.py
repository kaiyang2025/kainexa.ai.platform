#!/usr/bin/env python3
"""
API 통합 테스트
"""
import asyncio
import httpx
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

async def test_api():
    """API 테스트"""
    
    print("="*60)
    print("Kainexa API 통합 테스트")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        
        # 1. 헬스체크
        print("\n1️⃣ 헬스체크...")
        response = await client.get(f"{API_BASE}/api/v1/health/full")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ 시스템 상태: {data['services']}")
        else:
            print(f"   ❌ 헬스체크 실패: {response.status_code}")
        
        # 2. 로그인
        print("\n2️⃣ 로그인...")
        response = await client.post(
            f"{API_BASE}/api/v1/login",
            json={"username": "admin", "password": "admin"}
        )
        if response.status_code == 200:
            login_data = response.json()
            print(f"   ✅ 로그인 성공: {login_data['username']}")
            session_token = login_data['session']['session_token']
        else:
            print(f"   ❌ 로그인 실패")
            session_token = None
        
        # 3. 채팅 테스트
        print("\n3️⃣ 채팅 테스트...")
        response = await client.post(
            f"{API_BASE}/api/v1/chat",
            json={"message": "스마트 팩토리의 OEE를 개선하는 방법은?"}
        )
        if response.status_code == 200:
            chat_data = response.json()
            print(f"   ✅ 응답: {chat_data['response'][:100]}...")
            conversation_id = chat_data['conversation_id']
        else:
            print(f"   ❌ 채팅 실패")
            conversation_id = None
        
        # 4. 문서 검색
        print("\n4️⃣ 문서 검색...")
        response = await client.get(
            f"{API_BASE}/api/v1/documents/search",
            params={"query": "품질 관리", "limit": 3}
        )
        if response.status_code == 200:
            search_data = response.json()
            print(f"   ✅ 검색 결과: {search_data['count']}개")
        else:
            print(f"   ❌ 검색 실패")
        
        # 5. 생산 모니터링 시나리오
        print("\n5️⃣ 생산 모니터링 시나리오...")
        response = await client.post(
            f"{API_BASE}/api/v1/scenarios/production",
            params={"query": "어제 생산 현황"}
        )
        if response.status_code == 200:
            prod_data = response.json()
            print(f"   ✅ 달성률: {prod_data['data']['total']['achievement_rate']}%")
            print(f"   ✅ 이슈: {len(prod_data['issues'])}개")
        else:
            print(f"   ❌ 시나리오 실패")
        
        # 6. 예측적 유지보수 시나리오
        print("\n6️⃣ 예측적 유지보수 시나리오...")
        response = await client.post(
            f"{API_BASE}/api/v1/scenarios/maintenance",
            params={"equipment_id": "CNC_007"}
        )
        if response.status_code == 200:
            maint_data = response.json()
            print(f"   ✅ 고장 확률: {maint_data['failure_probability']}%")
            print(f"   ✅ 예상 시점: {maint_data['predicted_failure_time']}")
        else:
            print(f"   ❌ 시나리오 실패")
        
        # 7. 품질 관리 시나리오
        print("\n7️⃣ 품질 관리 시나리오...")
        response = await client.post(f"{API_BASE}/api/v1/scenarios/quality")
        if response.status_code == 200:
            quality_data = response.json()
            print(f"   ✅ 불량률: {quality_data['summary']['defect_rate']}%")
            print(f"   ✅ Cpk: {quality_data['summary']['cpk']}")
        else:
            print(f"   ❌ 시나리오 실패")
        
        # 8. 대화 기록 조회
        if conversation_id:
            print("\n8️⃣ 대화 기록 조회...")
            response = await client.get(
                f"{API_BASE}/api/v1/conversations/{conversation_id}/history"
            )
            if response.status_code == 200:
                history_data = response.json()
                print(f"   ✅ 메시지 수: {len(history_data['messages'])}개")
            else:
                print(f"   ❌ 기록 조회 실패")
    
    print("\n" + "="*60)
    print("✅ API 테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_api())