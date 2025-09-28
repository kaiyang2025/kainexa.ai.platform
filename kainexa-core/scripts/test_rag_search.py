# scripts/test_rag_search.py 생성
#!/usr/bin/env python3
"""
RAG 검색 테스트
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.governance.rag_pipeline import RAGGovernance, AccessLevel

async def test_rag_search():
    """RAG 검색 테스트"""
    
    print("="*60)
    print("RAG 검색 테스트")
    print("="*60)
    
    # RAG 시스템 초기화
    rag = RAGGovernance(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="kainexa_knowledge"
    )
    
    # 테스트 쿼리들
    test_queries = [
        {
            "query": "OEE를 개선하는 방법",
            "expected": "스마트 팩토리"
        },
        {
            "query": "예측적 유지보수란?",
            "expected": "IoT 센서"
        },
        {
            "query": "품질 검사 종류",
            "expected": "수입 검사"
        },
        {
            "query": "생산 계획 수립 방법",
            "expected": "MRP"
        },
        {
            "query": "불량률 관리",
            "expected": "파레토"
        }
    ]
    
    # 각 쿼리 테스트
    for i, test in enumerate(test_queries, 1):
        print(f"\n{i}. 쿼리: '{test['query']}'")
        print("-" * 40)
        
        # 검색 실행
        results = await rag.retrieve(
            query=test['query'],
            k=3,
            user_access_level=AccessLevel.INTERNAL
        )
        
        if results:
            print(f"   검색 결과: {len(results)}개")
            
            for j, result in enumerate(results[:2], 1):
                print(f"\n   [{j}] 출처: {result.get('source', 'Unknown')}")
                print(f"       점수: {result.get('score', 0):.3f}")
                print(f"       내용: {result.get('text', '')[:150]}...")
                
                # 예상 키워드 확인
                if test['expected'] in result.get('text', ''):
                    print(f"       ✅ 예상 키워드 포함: '{test['expected']}'")
        else:
            print("   ❌ 검색 결과 없음")
    
    # 거버넌스 리포트
    print("\n" + "="*60)
    print("거버넌스 리포트")
    print("="*60)
    
    report = rag.get_governance_report()
    print(f"정책: {report['policies']}")
    print(f"상태: {report['governance_status']}")
    
    print("\n✅ RAG 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(test_rag_search())