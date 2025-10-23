# src/scenarios/predictive_maintenance.py 생성
"""
시나리오 2: 예측적 유지보수
"""
import asyncio
import os
from time import perf_counter
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List, Optional

# 무거운 의존성은 요청 시점에만 import (부팅 속도/안정성)
# LLM은 사용 시점에 lazy import 합니다. RAG도 core 네임스페이스로 수정.
from src.core.governance.rag_pipeline import RAGPipeline, AccessLevel

class PredictiveMaintenanceAgent:
    """예측적 유지보수 AI 에이전트"""

    def __init__(self, rag: Optional[RAGPipeline] = None, llm=None):
        # RAG/LLM은 주입 가능. LLM은 lazy import로 지연 로드.
        self.rag = rag or RAGPipeline()
        self.llm = llm
        self.equipment_history = self._load_equipment_history()
        
    def _load_equipment_history(self) -> Dict:
        """설비 이력 로드"""
        return {
            "CNC_007": {
                "install_date": "2020-03-15",
                "last_maintenance": "2024-10-15",
                "operating_hours": 18500,
                "failure_history": [
                    {"date": "2023-05-20", "cause": "베어링 마모", "downtime": 240},
                    {"date": "2024-02-10", "cause": "스핀들 이상", "downtime": 180}
                ],
                "current_sensors": {
                    "vibration": [2.1, 2.3, 2.5, 3.1, 3.8, 4.2],  # 최근 6일
                    "temperature": [65, 66, 68, 71, 75, 78],
                    "current": [45, 46, 46, 48, 51, 53],
                    "noise_db": [72, 73, 73, 75, 78, 82]
                }
            }
        }
    
    async def predict_failure(self, equipment_id: str) -> Dict[str, Any]:
        """고장 예측"""
        
        t0 = perf_counter()
        print(f"🔮 설비 {equipment_id} 고장 예측 분석...")
        
        # 설비 데이터 가져오기
        equipment = self.equipment_history[equipment_id]
        sensors = equipment['current_sensors']
        
        # 1. 센서 데이터 분석
        analysis = self._analyze_sensor_trends(sensors)
        
        # 2. 고장 확률 계산
        failure_probability = self._calculate_failure_probability(analysis)
        
        # 3. 보고서 생성 (기본: 템플릿 초고속, 필요 시 LLM 경로 토글)
        use_llm = os.getenv("KXN_USE_LLM_MAINT", os.getenv("KXN_USE_LLM_REPORT", "0")) == "1"
        if not use_llm:
            response = self._render_report_korean(equipment_id, equipment, sensors, analysis, failure_probability)
        else:            
            # 필요할 때만 LLM 로드 (torch/transformers 미설치 환경 보호)
            if self.llm is None:
                try:
                    from src.core.models.solar_llm import SolarLLM
                    self.llm = SolarLLM()
                except Exception as e:
                    # LLM 가용하지 않으면 템플릿으로 폴백
                    response = self._render_report_korean(equipment_id, equipment, sensors, analysis, failure_probability)
                    return self._build_result(equipment_id, analysis, failure_probability, response)
            self.llm.load()
            
            prompt = f"""
설비 ID: {equipment_id}
운영 시간: {equipment['operating_hours']}시간
마지막 정비: {equipment['last_maintenance']}

센서 데이터 트렌드:
- 진동: {analysis['vibration']['trend']} (현재: {sensors['vibration'][-1]})
- 온도: {analysis['temperature']['trend']} (현재: {sensors['temperature'][-1]}°C)
- 전류: {analysis['current']['trend']} (현재: {sensors['current'][-1]}A)
- 소음: {analysis['noise_db']['trend']} (현재: {sensors['noise_db'][-1]}dB)

고장 확률: {failure_probability}%

과거 유사 패턴(참고만, 결과는 한국어로만 기술):
- 2023년 베어링 마모 시: 진동 4.0, 온도 76°C
- 2024년 스핀들 이상 시: 진동 3.5, 소음 80dB

위 정보를 바탕으로 **한국어(한글)로만** 간결한 예지보전 보고서를 작성하세요.
필수 섹션: 고장 징후/예상 시점/권장 조치/예방 정비 계획.
숫자는 천 단위 쉼표, 백분율은 % 표기.
"""
            response = self.llm.generate(
                prompt,
                max_new_tokens=448,   # 속도 절충
                do_sample=False,      # 그리디(안정/속도)
                ko_only=True          # 한자/중문 토큰 금지
            )
        
        # 4. 정비 일정 제안
        maintenance_schedule = self._suggest_maintenance_schedule(
            failure_probability,
            equipment['last_maintenance']
        )
        
        
        duration_ms = int((perf_counter() - t0) * 1000)
        return {
            "equipment_id": equipment_id,
            "timestamp": datetime.now().isoformat(),
            "sensor_analysis": analysis,
            "failure_probability": failure_probability,
            "predicted_failure_time": self._estimate_failure_time(failure_probability),
            "ai_analysis": self._normalize_spacing(response),
            "maintenance_schedule": maintenance_schedule,
            "estimated_downtime": self._estimate_downtime(analysis),
            "spare_parts": self._recommend_spare_parts(analysis),
            "duration_ms": duration_ms
        }
    
    
    def _build_result(self, equipment_id: str, analysis: Dict[str, Any],
                      failure_probability: float, response: str) -> Dict[str, Any]:
        return {
            "equipment_id": equipment_id,
            "timestamp": datetime.now().isoformat(),
            "sensor_analysis": analysis,
            "failure_probability": failure_probability,
            "predicted_failure_time": self._estimate_failure_time(failure_probability),
            "ai_analysis": self._normalize_spacing(response),
            "maintenance_schedule": self._suggest_maintenance_schedule(
                failure_probability, self.equipment_history[equipment_id]['last_maintenance']
            ),
            "estimated_downtime": self._estimate_downtime(analysis),
            "spare_parts": self._recommend_spare_parts(analysis),
            "duration_ms": 0
        }
        
    def _analyze_sensor_trends(self, sensors: Dict) -> Dict:
        """센서 트렌드 분석"""
        analysis = {}
        
        for sensor_name, values in sensors.items():
            values_array = np.array(values)
            
            # 선형 회귀로 트렌드 계산
            x = np.arange(len(values))
            slope = np.polyfit(x, values_array, 1)[0]
            
            # 변화율 계산
            change_rate = ((values[-1] - values[0]) / values[0]) * 100
            
            # 임계값 초과 확인
            thresholds = {
                'vibration': 3.5,
                'temperature': 70,
                'current': 50,
                'noise_db': 75
            }
            
            over_threshold = values[-1] > thresholds.get(sensor_name, float('inf'))
            
            analysis[sensor_name] = {
                'current': values[-1],
                'trend': 'increasing' if slope > 0 else 'stable',
                'change_rate': round(change_rate, 1),
                'slope': round(slope, 2),
                'over_threshold': over_threshold
            }
        
        return analysis
    
    def _calculate_failure_probability(self, analysis: Dict) -> float:
        """고장 확률 계산"""
        probability = 0.0
        
        # 각 센서별 가중치
        weights = {
            'vibration': 0.4,
            'temperature': 0.3,
            'current': 0.2,
            'noise_db': 0.1
        }
        
        for sensor, data in analysis.items():
            if sensor in weights:
                # 임계값 초과시 확률 증가
                if data['over_threshold']:
                    probability += weights[sensor] * 50
                
                # 증가 트렌드시 확률 증가
                if data['trend'] == 'increasing':
                    probability += weights[sensor] * 30 * (data['change_rate'] / 100)
        
        return min(round(probability, 1), 95.0)
    
    def _estimate_failure_time(self, probability: float) -> str:
        """예상 고장 시점"""
        if probability > 80:
            return "24-48시간 이내"
        elif probability > 60:
            return "48-72시간 이내"
        elif probability > 40:
            return "1주일 이내"
        else:
            return "2주 이상"
    
    def _suggest_maintenance_schedule(self, probability: float, last_maintenance: str) -> Dict:
        """정비 일정 제안"""
        if probability > 70:
            return {
                "urgency": "긴급",
                "recommended_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                "type": "예방 정비",
                "estimated_duration": "2-3시간"
            }
        elif probability > 50:
            return {
                "urgency": "높음",
                "recommended_date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                "type": "점검 및 부품 교체",
                "estimated_duration": "1-2시간"
            }
        else:
            return {
                "urgency": "보통",
                "recommended_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "type": "정기 점검",
                "estimated_duration": "30분-1시간"
            }
    
    def _estimate_downtime(self, analysis: Dict) -> str:
        """예상 정지 시간"""
        critical_sensors = sum(1 for s in analysis.values() if s['over_threshold'])
        
        if critical_sensors >= 3:
            return "4-6시간"
        elif critical_sensors >= 2:
            return "2-4시간"
        else:
            return "1-2시간"
    
    def _recommend_spare_parts(self, analysis: Dict) -> List[str]:
        """예비 부품 추천"""
        parts = []
        
        if analysis['vibration']['over_threshold']:
            parts.append("스핀들 베어링 세트")
        
        if analysis['temperature']['over_threshold']:
            parts.append("냉각 시스템 부품")
        
        if analysis['current']['over_threshold']:
            parts.append("모터 드라이브")
        
        if analysis['noise_db']['over_threshold']:
            parts.append("방진 패드")
        
        return parts if parts else ["정기 소모품"]
    
    
    
    # ───────────── 템플릿 기반 한국어 보고서(초고속) ─────────────
    def _render_report_korean(
        self,
        equipment_id: str,
        equipment: Dict[str, Any],
        sensors: Dict[str, List[float]],
        analysis: Dict[str, Any],
        failure_probability: float
    ) -> str:
        def pct(x): return f"{x:.1f}%"
        vib = analysis["vibration"]; tmp = analysis["temperature"]
        cur = analysis["current"];   noi = analysis["noise_db"]
        lines = [
            f"- 진동: 현재 {sensors['vibration'][-1]} (추세: {vib['trend']}, 변화율 {pct(vib['change_rate'])})",
            f"- 온도: 현재 {sensors['temperature'][-1]}°C (추세: {tmp['trend']}, 변화율 {pct(tmp['change_rate'])})",
            f"- 전류: 현재 {sensors['current'][-1]}A (추세: {cur['trend']}, 변화율 {pct(cur['change_rate'])})",
            f"- 소음: 현재 {sensors['noise_db'][-1]}dB (추세: {noi['trend']}, 변화율 {pct(noi['change_rate'])})",
        ]
        issue_flags = []
        for k, label in [("vibration","진동"),("temperature","온도"),("current","전류"),("noise_db","소음")]:
            if analysis[k]["over_threshold"]:
                issue_flags.append(f"- {label}: 임계 초과")
        md = [
            f"# 예지보전 보고서 ({equipment_id})",
            f"- 운영 시간: **{equipment['operating_hours']:,}** 시간",
            f"- 마지막 정비: **{equipment['last_maintenance']}**",
            "\n## 센서 트렌드",
            *lines,
            f"\n## 고장 예측",
            f"- 예상 확률: **{failure_probability:.1f}%**",
            f"- 예상 시점: **{self._estimate_failure_time(failure_probability)}**",
            f"- 예상 정지 시간: **{self._estimate_downtime(analysis)}**",
            "\n## 주요 징후",
            *(issue_flags or ["- 임계 초과 징후 없음"]),
            "\n## 권장 조치",
            "- 임계 초과/상승 추세 센서 원인 점검(베어링/냉각/전장/방진)",
            "- 예비 부품 확보 및 취약 부품 선교체 검토",
            "- 조건부 예방 정비 일정 수립 및 가동 중 점검 강화",
            "\n## 예방 정비 계획(제안)",
            f"- 권장 일정: **{self._suggest_maintenance_schedule(failure_probability, equipment['last_maintenance'])['recommended_date']}**",
            f"- 유형/소요: **{self._suggest_maintenance_schedule(failure_probability, equipment['last_maintenance'])['type']}**, **{self._suggest_maintenance_schedule(failure_probability, equipment['last_maintenance'])['estimated_duration']}**",
        ]
        return "\n".join(md)

    def _normalize_spacing(self, text: str) -> str:
        """과다 공백/붙여쓰기 최소 정리"""
        import re
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\n[ \t]+', '\n', text)
        text = re.sub(r'\s+([,.:;)\]%])', r'\1', text)
        text = re.sub(r'([,.:%])(?=[^\s\n])', r'\1 ', text)
        return text.strip()

async def run_scenario():
    """시나리오 실행"""
    print("="*60)
    print("시나리오 2: 예측적 유지보수")
    print("="*60)
    
    agent = PredictiveMaintenanceAgent()
    
    # CNC 설비 분석
    result = await agent.predict_failure("CNC_007")
    
    print(f"\n🏭 설비 ID: {result['equipment_id']}")
    print(f"⏰ 분석 시간: {result['timestamp']}")
    
    print(f"\n📊 센서 분석:")
    for sensor, data in result['sensor_analysis'].items():
        print(f"   {sensor}: {data['current']} ({data['trend']}, "
              f"변화율: {data['change_rate']}%)")
    
    print(f"\n⚠️ 고장 예측:")
    print(f"   확률: {result['failure_probability']}%")
    print(f"   예상 시점: {result['predicted_failure_time']}")
    print(f"   예상 정지시간: {result['estimated_downtime']}")
    
    print(f"\n🔧 정비 계획:")
    schedule = result['maintenance_schedule']
    print(f"   긴급도: {schedule['urgency']}")
    print(f"   권장 날짜: {schedule['recommended_date']}")
    print(f"   정비 유형: {schedule['type']}")
    print(f"   소요 시간: {schedule['estimated_duration']}")
    
    print(f"\n📦 필요 부품:")
    for part in result['spare_parts']:
        print(f"   - {part}")
    
    print(f"\n🤖 AI 분석:")
    print(result['ai_analysis'][:500])
    
    print("\n✅ 시나리오 완료!")

if __name__ == "__main__":
    asyncio.run(run_scenario())
