# src/scenarios/manufacturing_demo.py
"""제조업 시나리오 데모"""
import asyncio
from datetime import datetime
from typing import Dict, Any

from src.orchestration.dsl_parser import DSLParser
from src.orchestration.graph_executor import GraphExecutor
from src.auth.mcp_permissions import MCPAuthManager, Role

class ManufacturingScenario:
    """제조업 AI Agent 시나리오"""
    
    def __init__(self):
        self.auth_manager = MCPAuthManager("secret")
        self.graph_executor = GraphExecutor()
        
    async def run_production_monitoring(self):
        """생산 모니터링 시나리오"""
        
        print("\n" + "="*60)
        print("🏭 생산 관리 AI Agent - 실시간 모니터링")
        print("="*60)
        
        # 김부장 인증
        token = self.auth_manager.create_token(
            user_id="kim_manager",
            role=Role.AGENT,
            metadata={'name': '김부장', 'dept': '생산관리팀'}
        )
        
        # 워크플로우 정의
        workflow = """
        name: production_monitoring
        graph:
          - step: authenticate
            type: auth_check
          - step: retrieve_production_data
            type: retrieve_knowledge
            params:
              query: "어제 밤사 생산 현황"
              filters: 
                time_range: "last_24h"
                data_type: "production"
          - step: analyze_issues
            type: llm_generate
            params:
              prompt_template: |
                생산 데이터 분석:
                {retrieved_knowledge}
                
                주요 이슈와 제안사항을 포함한 보고서를 작성하세요.
          - step: format_response
            type: response_postprocess
            params:
              target_honorific: "hamnida"
        """
        
        # 실행
        parser = DSLParser()
        graph = parser.parse_yaml(workflow)
        
        context = {
            'input_text': "어제 밤사 생산 현황 보고해줘",
            'user_token': token,
            'user_role': 'agent'
        }
        
        result = await self.graph_executor.execute_graph(graph, context)
        
        print("\n📊 AI Agent 응답:")
        print("-" * 40)
        print(result.get('final_response', 'No response'))
        
    async def run_predictive_maintenance(self):
        """예측적 유지보수 시나리오"""
        
        print("\n" + "="*60)
        print("⚙️ 예측적 유지보수 AI Agent")
        print("="*60)
        
        # 실시간 설비 데이터 시뮬레이션
        equipment_data = {
            'equipment_id': 'CNC_007',
            'vibration': 4.2,  # 정상: 2.0-3.0
            'temperature': 78,  # 정상: 60-70
            'operating_hours': 18500,
            'last_maintenance': '2024-10-15'
        }
        
        print(f"\n🔍 설비 모니터링: {equipment_data['equipment_id']}")
        print(f"   진동: {equipment_data['vibration']} (임계값 초과)")
        print(f"   온도: {equipment_data['temperature']}°C")
        
        # AI 분석 시뮬레이션
        await asyncio.sleep(1)
        
        print("\n🚨 AI 분석 결과:")
        print("   고장 예측: 72시간 내 78% 확률")
        print("   원인: 스핀들 베어링 마모")
        print("   권장조치: 즉시 감속 운전, 내일 오전 베어링 교체")
        
    async def run_quality_analysis(self):
        """품질 분석 시나리오"""
        
        print("\n" + "="*60)
        print("📈 품질 관리 AI Agent")
        print("="*60)
        
        # 품질 데이터
        quality_metrics = {
            'defect_rate': 0.28,
            'cpk': 1.52,
            'customer_claims': 1,
            'ai_detected_cracks': 3
        }
        
        print(f"\n📊 주간 품질 지표:")
        print(f"   불량률: {quality_metrics['defect_rate']}%")
        print(f"   Cpk: {quality_metrics['cpk']}")
        print(f"   고객 클레임: {quality_metrics['customer_claims']}건")
        
        print(f"\n🤖 AI 비전 검사:")
        print(f"   미세 크랙 {quality_metrics['ai_detected_cracks']}건 사전 감지")
        print(f"   예상 클레임 방지 효과: 8,500만원")

async def main():
    """메인 실행"""
    scenario = ManufacturingScenario()
    
    print("\n" + "🚀 "*20)
    print("Kainexa AI Agent Platform - 제조업 데모")
    print("🚀 "*20)
    
    # 시나리오 실행
    await scenario.run_production_monitoring()
    await asyncio.sleep(2)
    
    await scenario.run_predictive_maintenance()
    await asyncio.sleep(2)
    
    await scenario.run_quality_analysis()
    
    print("\n" + "="*60)
    print("✅ 시나리오 실행 완료!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())