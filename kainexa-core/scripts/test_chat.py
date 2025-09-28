#!/usr/bin/env python3
"""채팅 API 테스트"""
import asyncio
import httpx
import json

async def test_chat():
    url = "http://localhost:8000/api/v1/chat"
    
    questions = [
        "스마트 팩토리에서 OEE를 개선하는 방법은?",
        "예측적 유지보수란 무엇인가요?",
        "품질 관리를 어떻게 개선할 수 있나요?"
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for q in questions:
            print(f"\n질문: {q}")
            try:
                response = await client.post(
                    url,
                    json={"message": q, "user_email": "test@kainexa.local"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"응답: {data['response'][:300]}...")
                else:
                    print(f"에러 코드: {response.status_code}")
                    print(f"에러 내용: {response.text}")
                    
            except Exception as e:
                print(f"요청 실패: {e}")

if __name__ == "__main__":
    asyncio.run(test_chat())