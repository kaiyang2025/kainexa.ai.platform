# src/scenarios/production_monitoring.py 생성
"""
시나리오 1: 실시간 생산 모니터링 및 이상 감지
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os, re, json
import random

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
        
        # ── 빠른 경로(기본): LLM 없이 한국어 보고서 템플릿 생성 ─────────────────
        use_llm = os.getenv("KXN_USE_LLM_REPORT", "0") == "1"
        if not use_llm:
            response = self._render_report_korean(data, issues)
        else:
            # ── 느리지만 유연한 경로: LLM 사용(그리디 + 한국어 강제) ───────────────
            prompt = f"""
사용자 질문: {user_query}

생산 데이터:
- 계획 생산량: {data['total']['planned']:,}개
- 실제 생산량: {data['total']['actual']:,}개
- 달성률: {data['total']['achievement_rate']:.1f}%
- 불량품: {data['total']['defects']:,}개 (불량률: {data['total'].get('defect_rate_pct', data['total'].get('defect_rate', 0.0)):.2f}%)

주요 이슈:
{json.dumps(issues, ensure_ascii=False, indent=2)}

관련 지식(요약만 참고, 외국어 표현은 모두 한국어로 바꿔 기술):
{rag_context}

반드시 한국어(한글)로만, 마크다운 섹션(생산 현황/주요 이슈/권장 조치)으로 간결히 작성하세요.
숫자는 천 단위 쉼표, 백분율은 %를 붙이세요. 외국어·한자·중국어 혼용 금지.
"""
            response = self.llm.generate(
                prompt,
                max_new_tokens=448,   # 속도↑
                do_sample=False,      # 그리디
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


    # ── 템플릿 기반 한국어 보고서(초고속) ─────────────────────────────────────────
    def _render_report_korean(self, data: Dict[str, Any], issues: list) -> str:
        t = data["total"]
        fnum = lambda n: f"{n:,}"
        lines_md = []
        for name, info in data["lines"].items():
            parts = [
                f"- **{name.replace('_',' ').title()}**: 계획 {fnum(info['planned'])}개, "
                f"실적 {fnum(info['actual'])}개, 불량 {fnum(info['defects'])}개, "
                f"OEE {info.get('oee','-')}%"
            ]
            if "downtime_min" in info:
                parts.append(f"(정지 {info['downtime_min']}분)")
            if "speed" in info:
                parts.append(f"(속도 {info['speed']}%)")
            lines_md.append(" ".join(parts))
        issues_md = []
        for it in issues:
            desc = f"- **{it['line']}**: {it['type']} (심각도: {it['severity']})"
            if it.get("details"):
                desc += f", {it['details']}"
            if it.get("cause"):
                desc += f", 원인: {it['cause']}"
            if it.get("impact"):
                desc += f", 영향: {it['impact']}"
            if it.get("target"):
                desc += f", 목표: {it['target']}"
            issues_md.append(desc)
        md = []
        md.append("# 생산 현황")
        md.append(f"- 계획 생산량: **{fnum(t['planned'])}**개")
        md.append(f"- 실제 생산량: **{fnum(t['actual'])}**개")
        ach = f"{t['achievement_rate']:.1f}%"
        dfr = t.get("defect_rate_pct", t.get("defect_rate", 0.0))
        md.append(f"- 달성률: **{ach}** / 불량률: **{dfr:.2f}%**")
        md.append("\n## 라인별 요약")
        md.extend(lines_md)
        md.append("\n# 주요 이슈")
        md.extend(issues_md if issues_md else ["- 보고된 이슈 없음"])
        md.append("\n# 권장 조치")
        recs = self._generate_recommendations(issues)
        md.extend([f"- {r}" for r in recs] or ["- 별도 조치 없음"])
        return "\n".join(md)

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
