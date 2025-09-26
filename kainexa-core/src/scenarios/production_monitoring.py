# src/scenarios/production_monitoring.py 생성
"""
시나리오 1: 실시간 생산 모니터링 및 이상 감지
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import re
import random
import json

from src.models.solar_llm import SolarLLM
from src.governance.rag_pipeline import RAGGovernance, AccessLevel


class ProductionMonitoringAgent:
    """생산 모니터링 AI 에이전트"""

    def __init__(self, rag: Optional[RAGGovernance] = None, llm: Optional[SolarLLM] = None):
        # 라우트에서 주입되면 사용, 아니면 내부 생성(하위호환)
        self.llm = llm or SolarLLM()
        self.rag = rag or RAGGovernance()
        self.production_data = self._simulate_production_data()
  
        
    def _simulate_production_data(self) -> Dict[str, Any]:
        """생산 데이터 시뮬레이션"""
        return {
            "timestamp": datetime.now(),
            "shift": "야간",
            "lines": {
                "line_1": {
                    "planned": 4000,
                    "actual": 3920,
                    "defects": 8,
                    "status": "normal",
                    "oee": 97.8
                },
                "line_2": {
                    "planned": 4000,
                    "actual": 3847,
                    "defects": 10,
                    "downtime_min": 15,
                    "issue": "금형 온도 이상",
                    "status": "warning",
                    "oee": 95.9
                },
                "line_3": {
                    "planned": 4000,
                    "actual": 3900,
                    "defects": 5,
                    "speed": 85,  # 정상 대비 %
                    "status": "slow",
                    "oee": 97.1
                }
            },
            "total": {
                "planned": 12000,
                "actual": 11667,
                "defects": 23,
                "achievement_rate": 97.2,
                "defect_rate": 0.19
            }
        }
    
    async def analyze_production(self, user_query: str) -> Dict[str, Any]:
        """생산 현황 분석"""
        
        print("🔍 생산 데이터 분석 중...")
        
        # 1. 데이터 수집
        data = self.production_data
        
        # 2. 이상 감지
        issues = self._detect_anomalies(data)
        
        # 3. RAG에서 관련 지식 검색
        rag_context = ""
        if issues:
            search_query = " ".join([issue['type'] for issue in issues])
            rag_results = await self.rag.retrieve(
                query=search_query,
                k=3,
                user_access_level=AccessLevel.INTERNAL
            )
            
            if rag_results:
                rag_context = "\n".join([r['text'][:200] for r in rag_results])
        
        # 4. LLM으로 보고서 생성
        self.llm.load()
        
        prompt = f"""
사용자 질문: {user_query}
 
 생산 데이터:
 - 계획 생산량: {data['total']['planned']}개
 - 실제 생산량: {data['total']['actual']}개
 - 달성률: {data['total']['achievement_rate']}%
 - 불량품: {data['total']['defects']}개 (불량률: {data['total']['defect_rate']}%)
 
 주요 이슈:
 {json.dumps(issues, ensure_ascii=False, indent=2)}
 
 관련 지식:
 {rag_context}

위 정보를 바탕으로 생산 관리자에게 보고할 내용을 **한국어(한글)로만** 작성하세요.
금지: 한자/중국어/영어 혼용, 의미 없는 붙여쓰기.
형식: 마크다운 제목과 목록으로 구성하세요. 제목은 다음 3개 섹션을 포함합니다.
- 생산 현황
- 주요 이슈
- 권장 조치
표기 규칙:
- 숫자는 천 단위 쉼표 사용(예: 11,667)
- 백분율은 % 기호 사용(예: 97.2%)
"""
        # 1차 생성: 샘플링 비활성화(그리디)로 안정적인 문장 생성
        response = self.llm.generate(
            prompt,
            max_new_tokens=768,
            temperature=0.2,
            top_p=1.0,
            do_sample=False,
        )

        response = self._normalize_spacing(response)
        # 2차: 항상 재작성(한국어만, 띄어쓰기 교정, 문장 부호 표준화, 마크다운 구성)
        rewrite_prompt = (
            "아래 초안을 자연스러운 한국어 보고서로만 다시 작성하세요. "
            "한자/중국어/영어 혼용 금지, 올바른 띄어쓰기 적용, 문장 부호 표준화, "
            "마크다운 제목(생산 현황/주요 이슈/권장 조치)과 목록 형식을 유지하세요:\n\n"
            + response
        )
        response = self.llm.generate(
            rewrite_prompt,
            max_new_tokens=640,
            temperature=0.2,
            top_p=1.0,
            do_sample=False,
        )
        response = self._normalize_spacing(response)
       
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "issues": issues,
            "report": response,
            "recommendations": self._generate_recommendations(issues)
        }
               
    
    # --- helpers ---------------------------------------------------------
    def _normalize_spacing(self, text: str) -> str:
        """한글 단어 간 공백은 유지하고, 과다/이상 공백과 문장부호 주변 공백만 정리"""
        # 탭/다중 스페이스 축소
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # 줄 앞쪽 여백 제거
        text = re.sub(r'\n[ \t]+', '\n', text)
        # 문장부호 앞 공백 제거
        text = re.sub(r'\s+([,.:;)\]%])', r'\1', text)
        # 문장부호 뒤 공백 확보(줄바꿈/공백/문장끝 제외)
        text = re.sub(r'([,.:%])(?=[^\s\n])', r'\1 ', text)
        return text.strip()

    def _contains_chinese(self, text: str) -> bool:
        """중국어 한자 포함 여부"""
        return bool(re.search(r'[\u4E00-\u9FFF]', text))

    def _needs_spacing_fix(self, text: str) -> bool:
        """한글 대비 공백이 비정상적으로 적으면 True (띄어쓰기 붕괴 감지)"""
        hangul = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')
        spaces = text.count(' ')
        # 한글이 충분히 많은데도 공백 비율이 아주 낮으면 재작성 유도
        return hangul >= 80 and (spaces / max(hangul, 1)) < 0.05

    def _korean_ratio(self, text: str) -> float:
        """텍스트 중 한글(가~힣) 비중 계산"""
        chars = [c for c in text if not c.isspace()]
        if not chars:
            return 1.0
        hangul = sum(1 for c in chars if '\uAC00' <= c <= '\uD7A3')
        return hangul / len(chars)
    
    def _detect_anomalies(self, data: Dict) -> List[Dict]:
        """이상 감지"""
        issues = []
        
        for line_id, line_data in data['lines'].items():
            # 가동 중단 확인
            if 'downtime_min' in line_data and line_data['downtime_min'] > 10:
                issues.append({
                    "line": line_id,
                    "type": "downtime",
                    "severity": "high",
                    "details": f"{line_data['downtime_min']}분 정지",
                    "cause": line_data.get('issue', 'Unknown')
                })
            
            # 속도 저하 확인
            if 'speed' in line_data and line_data['speed'] < 90:
                issues.append({
                    "line": line_id,
                    "type": "speed_degradation",
                    "severity": "medium",
                    "details": f"속도 {line_data['speed']}%",
                    "impact": "생산량 감소"
                })
            
            # OEE 저하 확인
            if line_data.get('oee', 100) < 96:
                issues.append({
                    "line": line_id,
                    "type": "low_oee",
                    "severity": "low",
                    "details": f"OEE {line_data['oee']}%",
                    "target": "96%"
                })
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """권장 조치사항 생성"""
        recommendations = []
        
        for issue in issues:
            if issue['type'] == 'downtime':
                recommendations.append(
                    f"{issue['line']}: {issue['cause']} 점검 필요"
                )
            elif issue['type'] == 'speed_degradation':
                recommendations.append(
                    f"{issue['line']}: 설비 점검 및 속도 조정 필요"
                )
            elif issue['type'] == 'low_oee':
                recommendations.append(
                    f"{issue['line']}: OEE 개선 방안 검토 필요"
                )
        
        return recommendations

async def run_scenario():
    """시나리오 실행"""
    print("="*60)
    print("시나리오 1: 생산 모니터링 및 이상 감지")
    print("="*60)
    
    agent = ProductionMonitoringAgent()
    
    # 김부장의 질문
    result = await agent.analyze_production(
        "어제 밤사 생산 현황 보고해줘"
    )
    
    print(f"\n⏰ 보고 시간: {result['timestamp']}")
    print(f"\n📊 생산 실적:")
    print(f"   달성률: {result['data']['total']['achievement_rate']}%")
    print(f"   불량률: {result['data']['total']['defect_rate']}%")
    
    print(f"\n⚠️ 감지된 이슈: {len(result['issues'])}개")
    for issue in result['issues']:
        print(f"   - {issue['line']}: {issue['details']} ({issue['severity']})")
    
    print(f"\n📝 AI 보고서:")
    print(result['report'][:500])
    
    print(f"\n💡 권장 조치:")
    for rec in result['recommendations']:
        print(f"   - {rec}")
    
    print("\n✅ 시나리오 완료!")

if __name__ == "__main__":
    asyncio.run(run_scenario())
