#!/usr/bin/env python3
"""채팅 API 디버깅"""
import asyncio
import httpx
import json
import traceback

async def test_connection():
    """연결 테스트"""
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("API 연결 테스트")
    print("=" * 60)
    
    # 1. 기본 연결 테스트
    print("\n1. 루트 엔드포인트 테스트...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(base_url + "/")
            print(f"   상태 코드: {response.status_code}")
            if response.status_code == 200:
                print(f"   응답: {response.json()}")
    except Exception as e:
        print(f"   ❌ 연결 실패: {e}")
        print(f"   서버가 실행 중인지 확인하세요!")
        return False
    
    # 2. 헬스체크
    print("\n2. 헬스체크...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(base_url + "/api/v1/health/full")
            print(f"   상태 코드: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   LLM 상태: {data.get('services', {}).get('llm', {}).get('status', 'unknown')}")
    except Exception as e:
        print(f"   ❌ 헬스체크 실패: {e}")
    
    # 3. 채팅 테스트 (상세 에러)
    print("\n3. 채팅 API 테스트...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "message": "안녕하세요",
                "user_email": "test@kainexa.local"
            }
            print(f"   요청 페이로드: {payload}")
            
            response = await client.post(
                base_url + "/api/v1/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ 응답: {data.get('response', '')[:100]}...")
            else:
                print(f"   ❌ 에러 응답:")
                print(f"   헤더: {response.headers}")
                print(f"   내용: {response.text[:500]}")
                
    except httpx.ConnectError as e:
        print(f"   ❌ 연결 불가: {e}")
        print(f"   - 서버가 실행 중인지 확인")
        print(f"   - 포트 8000이 열려있는지 확인")
    except httpx.TimeoutException as e:
        print(f"   ❌ 타임아웃: {e}")
        print(f"   - 모델 로딩이 완료되었는지 확인")
    except Exception as e:
        print(f"   ❌ 예외: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    asyncio.run(test_connection())