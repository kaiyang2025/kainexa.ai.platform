# src/scenarios/quality_control.py 생성
"""
시나리오 3: 품질 관리 및 불량 분석
"""
import asyncio
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List, Optional
import random

from src.models.solar_llm import SolarLLM
from src.governance.rag_pipeline import RAGGovernance

class QualityControlAgent:
    """품질 관리 AI 에이전트"""

    def __init__(self, rag: Optional[RAGGovernance] = None, llm: Optional[SolarLLM] = None):
        self.rag = rag
        self.llm = llm or SolarLLM()
        self.quality_data = self._generate_quality_data()
        
    def _generate_quality_data(self) -> Dict:
        """품질 데이터 생성"""
        return {
            "period": "2024-11-20 ~ 2024-11-26",
            "production_volume": 50000,
            "inspected": 5000,
            "defects": {
                "total": 140,
                "types": {
                    "dimension": 59,  # 42%
                    "surface": 39,     # 28%
                    "assembly": 28,    # 20%
                    "others": 14       # 10%
                }
            },
            "metrics": {
                "defect_rate": 0.28,
                "cpk": 1.52,
                "sigma_level": 4.2,
                "customer_claims": 1,
                "rework_rate": 0.15
            },
            "by_line": {
                "line_1": {"defect_rate": 0.25, "cpk": 1.55},
                "line_2": {"defect_rate": 0.35, "cpk": 1.48},
                "line_3": {"defect_rate": 0.24, "cpk": 1.53}
            },
            "by_shift": {
                "day": {"defect_rate": 0.22},
                "evening": {"defect_rate": 0.28},
                "night": {"defect_rate": 0.34}
            },
            "ai_detection": {
                "micro_cracks": 3,
                "prevented_claims": 8500000  # 원
            }
        }
    
    async def analyze_quality(self) -> Dict[str, Any]:
        """품질 분석"""
        
        print("🔍 품질 데이터 분석 중...")
        
        data = self.quality_data
        
        # 1. 트렌드 분석
        trends = self._analyze_trends(data)
        
        # 2. 불량 원인 분석
        root_causes = self._analyze_root_causes(data)
        
        # 3. 패턴 발견
        patterns = self._find_patterns(data)
        
        # 4. LLM으로 종합 분석
        self.llm.load()
        
        prompt = f"""
품질 관리 주간 보고서 작성

기간: {data['period']}
생산량: {data['production_volume']:,}개
검사량: {data['inspected']:,}개

품질 지표:
- 불량률: {data['metrics']['defect_rate']}%
- Cpk: {data['metrics']['cpk']}
- 시그마 레벨: {data['metrics']['sigma_level']}
- 고객 클레임: {data['metrics']['customer_claims']}건

불량 유형 분석:
- 치수 불량: {data['defects']['types']['dimension']}개 (42%)
- 외관 불량: {data['defects']['types']['surface']}개 (28%)
- 조립 불량: {data['defects']['types']['assembly']}개 (20%)
- 기타: {data['defects']['types']['others']}개 (10%)

발견된 패턴:
{patterns}

근본 원인:
{root_causes}

AI 비전 검사 성과:
- 미세 크랙 {data['ai_detection']['micro_cracks']}건 사전 감지
- 예상 클레임 방지 효과: {data['ai_detection']['prevented_claims']:,}원

위 정보를 바탕으로 품질 개선 방안을 포함한 종합 보고서를 작성하세요.
"""
        
        response = self.llm.generate(
            prompt,
            max_new_tokens=500,
            temperature=0.3
        )
        
        # 5. 개선 계획 수립
        improvement_plan = self._create_improvement_plan(data, patterns, root_causes)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "period": data['period'],
            "summary": {
                "defect_rate": data['metrics']['defect_rate'],
                "cpk": data['metrics']['cpk'],
                "trend": trends['overall']
            },
            "patterns": patterns,
            "root_causes": root_causes,
            "ai_analysis": response,
            "improvement_plan": improvement_plan,
            "roi_estimation": self._estimate_roi(data)
        }
    
    def _analyze_trends(self, data: Dict) -> Dict:
        """트렌드 분석"""
        # 간단한 트렌드 시뮬레이션
        previous_defect_rate = 0.31
        current_defect_rate = data['metrics']['defect_rate']
        
        if current_defect_rate < previous_defect_rate:
            trend = "improving"
            change = round((previous_defect_rate - current_defect_rate) / previous_defect_rate * 100, 1)
        else:
            trend = "worsening"
            change = round((current_defect_rate - previous_defect_rate) / previous_defect_rate * 100, 1)
        
        return {
            "overall": trend,
            "change_percentage": change,
            "cpk_trend": "stable" if data['metrics']['cpk'] > 1.5 else "needs_improvement"
        }
    
    def _analyze_root_causes(self, data: Dict) -> List[Dict]:
        """근본 원인 분석"""
        causes = []
        
        # 치수 불량 원인
        if data['defects']['types']['dimension'] > 50:
            causes.append({
                "type": "dimension",
                "cause": "금형 마모",
                "evidence": "치수 불량 59개 (전체의 42%)",
                "solution": "금형 교체 또는 보정"
            })
        
        # 외관 불량 원인
        if data['defects']['types']['surface'] > 30:
            causes.append({
                "type": "surface",
                "cause": "도장 공정 온습도 관리 미흡",
                "evidence": "외관 불량 39개 (전체의 28%)",
                "solution": "공조 시스템 개선"
            })
        
        # 교대별 차이 원인
        night_rate = data['by_shift']['night']['defect_rate']
        day_rate = data['by_shift']['day']['defect_rate']
        
        if night_rate > day_rate * 1.3:
            causes.append({
                "type": "shift",
                "cause": "야간 근무 숙련도 차이",
                "evidence": f"야간 불량률 {night_rate}% vs 주간 {day_rate}%",
                "solution": "야간 근무자 추가 교육"
            })
        
        return causes
    
    def _find_patterns(self, data: Dict) -> List[str]:
        """패턴 발견"""
        patterns = []
        
        # 시간대별 패턴
        if data['by_shift']['night']['defect_rate'] > data['by_shift']['day']['defect_rate']:
            patterns.append("야간 교대 시 불량률 50% 높음")
        
        # 라인별 패턴
        worst_line = max(data['by_line'].items(), 
                        key=lambda x: x[1]['defect_rate'])
        if worst_line[1]['defect_rate'] > 0.3:
            patterns.append(f"{worst_line[0]} 불량률 {worst_line[1]['defect_rate']}%로 개선 필요")
        
        # 요일별 패턴 (시뮬레이션)
        patterns.append("화요일 오후 2-4시 불량률 40% 증가 (외부 온도 최고점)")
        
        return patterns
    
    def _create_improvement_plan(self, data: Dict, patterns: List, 
                                 causes: List) -> List[Dict]:
        """개선 계획 수립"""
        plan = []
        
        # 단기 계획 (1주)
        plan.append({
            "term": "단기",
            "duration": "1주",
            "actions": [
                "야간 근무자 집중 교육",
                "공조 시스템 점검 및 조정",
                "Line 2 금형 정밀 점검"
            ],
            "expected_improvement": "불량률 0.05% 감소"
        })
        
        # 중기 계획 (1개월)
        plan.append({
            "term": "중기",
            "duration": "1개월",
            "actions": [
                "AI 비전 검사 시스템 확대 적용",
                "통계적 공정 관리(SPC) 강화",
                "작업 표준서 업데이트"
            ],
            "expected_improvement": "불량률 0.1% 감소, Cpk 1.6 달성"
        })
        
        # 장기 계획 (3개월)
        plan.append({
            "term": "장기",
            "duration": "3개월",
            "actions": [
                "6시그마 프로젝트 실행",
                "자동화 설비 도입 검토",
                "품질 예측 AI 모델 구축"
            ],
            "expected_improvement": "불량률 0.15% 달성, 시그마 레벨 5.0"
        })
        
        return plan
    
    def _estimate_roi(self, data: Dict) -> Dict:
        """ROI 추정"""
        # 불량 감소에 따른 비용 절감
        defect_cost_per_unit = 50000  # 원
        current_defects = data['defects']['total']
        
        # 개선 후 예상
        improved_defect_rate = 0.15
        expected_defects = int(data['production_volume'] * improved_defect_rate / 100)
        
        saved_defects = current_defects - expected_defects
        cost_savings = saved_defects * defect_cost_per_unit
        
        # AI 시스템 효과
        ai_prevention = data['ai_detection']['prevented_claims']
        
        return {
            "monthly_savings": cost_savings,
            "ai_prevention_value": ai_prevention,
            "total_monthly_benefit": cost_savings + ai_prevention,
            "roi_percentage": 285  # 시뮬레이션 값
        }

async def run_scenario():
    """시나리오 실행"""
    print("="*60)
    print("시나리오 3: 품질 관리 및 불량 분석")
    print("="*60)
    
    agent = QualityControlAgent()
    
    # 품질 분석 실행
    result = await agent.analyze_quality()
    
    print(f"\n📅 분석 기간: {result['period']}")
    print(f"⏰ 보고 시간: {result['timestamp']}")
    
    print(f"\n📊 품질 지표:")
    print(f"   불량률: {result['summary']['defect_rate']}%")
    print(f"   Cpk: {result['summary']['cpk']}")
    print(f"   트렌드: {result['summary']['trend']}")
    
    print(f"\n🔍 발견된 패턴:")
    for pattern in result['patterns']:
        print(f"   - {pattern}")
    
    print(f"\n⚠️ 근본 원인:")
    for cause in result['root_causes']:
        print(f"   - {cause['type']}: {cause['cause']}")
        print(f"     해결방안: {cause['solution']}")
    
    print(f"\n📋 개선 계획:")
    for plan in result['improvement_plan']:
        print(f"   [{plan['term']}] {plan['duration']}:")
        for action in plan['actions']:
            print(f"     • {action}")
        print(f"     예상 효과: {plan['expected_improvement']}")
    
    print(f"\n💰 ROI 추정:")
    roi = result['roi_estimation']
    print(f"   월간 절감액: {roi['monthly_savings']:,}원")
    print(f"   AI 클레임 방지: {roi['ai_prevention_value']:,}원")
    print(f"   총 월간 효익: {roi['total_monthly_benefit']:,}원")
    print(f"   ROI: {roi['roi_percentage']}%")
    
    print(f"\n🤖 AI 분석:")
    print(result['ai_analysis'][:500])
    
    print("\n✅ 시나리오 완료!")
        
    
if __name__ == "__main__":
    asyncio.run(run_scenario())
