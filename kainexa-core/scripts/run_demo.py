# scripts/run_demo.py 생성
#!/usr/bin/env python3
"""
Kainexa AI Platform 통합 데모
"""
import asyncio
import httpx
import json
import time
from datetime import datetime
from httpx import Timeout, ReadTimeout, ConnectTimeout, HTTPStatusError
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

async def run_manufacturing_demo():
    """제조업 시나리오 데모"""
    
    console.print("\n[bold cyan]="*60)
    console.print("[bold yellow]🏭 Kainexa AI Platform - 제조업 통합 데모")
    console.print("[bold cyan]="*60)
    
    base_url = "http://localhost:8000"

    # 충분한 타임아웃(콜드 스타트/모델 로딩 대비)
    timeout = Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        # 간단한 재시도 래퍼
        async def post_with_retry(url, *, params=None, json=None, retries=3, backoff=1.5):
            last_exc = None
            for i in range(retries):
                try:
                    resp = await client.post(url, params=params, json=json)
                    # 5xx는 재시도
                    if resp.status_code >= 500:
                        raise HTTPStatusError("server error", request=resp.request, response=resp)
                    return resp
                except (ReadTimeout, ConnectTimeout, HTTPStatusError) as e:
                    last_exc = e
                    if i == retries - 1:
                        raise
                    await asyncio.sleep(backoff ** i)
            raise last_exc

        async def get_with_retry(url, *, params=None, retries=5, backoff=1.5):
            last_exc = None
            for i in range(retries):
                try:
                    resp = await client.get(url, params=params)
                    if resp.status_code >= 500:
                        raise HTTPStatusError("server error", request=resp.request, response=resp)
                    return resp
                except (ReadTimeout, ConnectTimeout, HTTPStatusError) as e:
                    last_exc = e
                    if i == retries - 1:
                        raise
                    await asyncio.sleep(backoff ** i)
            raise last_exc

        # 0. 서버 예열(헬스체크) : 모델/벡터스토어 초기화 시간 흡수
        console.print("\n[bold green]0. 서버 예열(헬스체크)")
        response = await get_with_retry("/api/v1/health/full")
        
        # 1. 시스템 상태 확인
        console.print("\n[bold green]1. 시스템 상태 확인")
        response = await client.get(f"{base_url}/api/v1/health/full")
        
        
        if response.status_code == 200:
            health = response.json()
            
            table = Table(title="시스템 상태")
            table.add_column("서비스", style="cyan")
            table.add_column("상태", style="green")
            
            for service, status in health['services'].items():
                status_text = "✅ 정상" if status.get('status') == 'healthy' else "⚠️ 확인필요"
                table.add_row(service, status_text)
            
            console.print(table)
        
        # 2. 생산 모니터링
        console.print("\n[bold green]2. 생산 모니터링 시나리오")
        console.print("[yellow]김부장: '어제 밤사 생산 현황 보고해줘'")
        
        response = await post_with_retry(
            "/api/v1/scenarios/production",
            params={"query": "어제 밤사 생산 현황 보고해줘"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            panel = Panel(
                f"""[bold]생산 현황[/bold]
📊 달성률: {result['data']['total']['achievement_rate']}%
🔧 불량률: {result['data']['total']['defect_rate']}%
⚠️  이슈: {len(result['issues'])}건 발견

[yellow]AI 분석:[/yellow]
{result['report'][:300]}...
""",
                title="생산 모니터링 결과",
                border_style="green"
            )
            console.print(panel)
        
        await asyncio.sleep(2)
        
        # 3. 예측적 유지보수
        console.print("\n[bold green]3. 예측적 유지보수 시나리오")
        console.print("[yellow]시스템: 'CNC_007 설비 상태 분석'")
        
        response = await post_with_retry(
            "/api/v1/scenarios/maintenance",
            params={"equipment_id": "CNC_007"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # 고장 확률에 따른 색상
            prob = result['failure_probability']
            color = "red" if prob > 70 else "yellow" if prob > 40 else "green"
            
            panel = Panel(
                f"""[bold]설비 상태 분석[/bold]
🏭 설비 ID: {result['equipment_id']}
⚠️  고장 확률: [{color}]{prob}%[/{color}]
📅 예상 고장 시점: {result['predicted_failure_time']}
🔧 권장 정비: {result['maintenance_schedule']['type']}
📆 정비 예정일: {result['maintenance_schedule']['recommended_date']}

[yellow]필요 부품:[/yellow]
{', '.join(result['spare_parts'])}
""",
                title="예측적 유지보수",
                border_style=color
            )
            console.print(panel)
        
        await asyncio.sleep(2)
        
        # 4. 품질 관리
        console.print("\n[bold green]4. 품질 관리 시나리오")
        console.print("[yellow]박차장: '이번 주 품질 트렌드 분석해줘'")
        
        response = await post_with_retry("/api/v1/scenarios/quality")
        
        if response.status_code == 200:
            result = response.json()
            
            panel = Panel(
                f"""[bold]품질 분석 결과[/bold]
📊 불량률: {result['summary']['defect_rate']}%
📈 Cpk: {result['summary']['cpk']}
📉 트렌드: {result['summary']['trend']}

[yellow]발견된 패턴:[/yellow]
{chr(10).join('• ' + p for p in result['patterns'][:3])}

[yellow]개선 계획:[/yellow]
{chr(10).join(f"• [{p['term']}] {p['actions'][0]}" for p in result['improvement_plan'][:2])}

💰 예상 ROI: {result['roi_estimation']['roi_percentage']}%
""",
                title="품질 관리",
                border_style="blue"
            )
            console.print(panel)
        
        await asyncio.sleep(2)
        
        # 5. 대화형 AI 테스트
        console.print("\n[bold green]5. 대화형 AI 어시스턴트")
        
        questions = [
            "스마트 팩토리에서 OEE를 개선하는 방법은?",
            "불량률이 증가했을 때 대응 방안은?",
            "예측적 유지보수의 장점은?"
        ]
        
        for q in questions[:1]:  # 시간 절약을 위해 1개만
            console.print(f"\n[yellow]질문: {q}")
            
            response = await post_with_retry(
                "/api/v1/chat",
                json={"message": q}
            )
            
            if response.status_code == 200:
                chat_result = response.json()
                console.print(f"[green]AI: {chat_result['response'][:300]}...")
    
    console.print("\n[bold cyan]="*60)
    console.print("[bold green]✅ 데모 완료! Kainexa AI Platform이 정상 작동 중입니다.")
    console.print("[bold cyan]="*60)

async def interactive_chat():
    """대화형 채팅 모드"""
    
    console.print("\n[bold cyan]💬 Kainexa AI 대화 모드")
    console.print("[dim]종료하려면 'exit' 또는 'quit'를 입력하세요.[/dim]\n")
    
    base_url = "http://localhost:8000"
    conversation_id = None
    
    timeout = Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        async def post_with_retry(url, *, params=None, json=None, retries=3, backoff=1.5):
            last_exc = None
            for i in range(retries):
                try:
                    resp = await client.post(url, params=params, json=json)
                    if resp.status_code >= 500:
                        raise HTTPStatusError("server error", request=resp.request, response=resp)
                    return resp
                except (ReadTimeout, ConnectTimeout, HTTPStatusError) as e:
                    last_exc = e
                    if i == retries - 1:
                        raise
                    await asyncio.sleep(backoff ** i)
            raise last_exc

        while True:
            # 사용자 입력
            user_input = console.input("[bold yellow]You>[/bold yellow] ")
            
            if user_input.lower() in ['exit', 'quit', '종료']:
                console.print("[dim]대화를 종료합니다...[/dim]")
                break
            
            # API 호출
            try:
                response = await post_with_retry(
                    "/api/v1/chat",
                    json={
                        "message": user_input,
                        "conversation_id": conversation_id
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    conversation_id = result.get('conversation_id')
                    
                    console.print(f"[bold green]AI>[/bold green] {result['response']}")
                    
                    if result.get('sources'):
                        console.print(f"[dim]출처: {', '.join(result['sources'])[:100]}[/dim]")
                else:
                    console.print(f"[red]오류: {response.status_code}[/red]")
                    
            except Exception as e:
                console.print(f"[red]오류: {e}[/red]")

def main():
    """메인 메뉴"""
    
    console.print("""
[bold cyan]🚀 Kainexa AI Platform - 실행 모드 선택[/bold cyan]

1. 제조업 시나리오 데모
2. 대화형 AI 채팅
3. 종료

""")
    
    choice = console.input("선택 [1-3]: ")
    
    if choice == "1":
        asyncio.run(run_manufacturing_demo())
    elif choice == "2":
        asyncio.run(interactive_chat())
    else:
        console.print("종료합니다.")

if __name__ == "__main__":
    # rich 설치 확인
    try:
        import rich
    except ImportError:
        print("Installing rich for better display...")
        import os
        os.system("pip install rich")
    
    main()