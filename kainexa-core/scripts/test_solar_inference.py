# scripts/test_solar_inference.py 생성
#!/usr/bin/env python3
"""
Solar LLM 추론 테스트
"""
import sys
import asyncio
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.solar_llm import SolarLLM

async def test_inference():
    """추론 테스트"""
    
    print("="*60)
    print("Solar-10.7B 추론 테스트")
    print("="*60)
    
    # 모델 초기화
    llm = SolarLLM(
        model_path="models/solar-10.7b",
        load_in_8bit=True  # 메모리 절약
    )
    
    # 모델 로드
    print("\n1️⃣ 모델 로딩...")
    llm.load()
    
    # 메모리 사용량 확인
    memory = llm.get_memory_usage()
    for gpu, stats in memory.items():
        print(f"   {gpu}: {stats['allocated_gb']}GB / {stats['total_gb']}GB ({stats['usage_percent']}%)")
    
    # 테스트 케이스
    test_cases = [
        {
            "name": "제조업 질문",
            "prompt": "스마트 팩토리에서 OEE를 향상시키는 방법을 설명해주세요.",
            "max_tokens": 256
        },
        {
            "name": "생산 분석",
            "prompt": "불량률이 3%에서 5%로 증가했습니다. 원인 분석과 대응 방안을 제시해주세요.",
            "max_tokens": 300
        },
        {
            "name": "한국어 대화",
            "prompt": "안녕하세요. 제조업 AI 어시스턴트로서 어떤 도움을 줄 수 있나요?",
            "max_tokens": 200
        }
    ]
    
    # 추론 실행
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ 테스트: {test_case['name']}")
        print(f"   프롬프트: {test_case['prompt'][:50]}...")
        
        response = llm.generate(
            test_case['prompt'],
            max_new_tokens=test_case['max_tokens'],
            temperature=0.7,
            stream=False  # True로 하면 실시간 스트리밍
        )
        
        print(f"   응답: {response[:200]}...")
        print(f"   전체 길이: {len(response)} 문자")
    
    print("\n" + "="*60)
    print("✅ 추론 테스트 완료!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_inference())