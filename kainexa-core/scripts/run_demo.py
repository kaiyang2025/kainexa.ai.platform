#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kainexa AI 데모 실행 스크립트 (비동기)
- 서버 헬스체크
- 생산/정비/품질 시나리오 호출
- 대화형 AI 어시스턴트 호출
특징:
  * httpx.AsyncClient + 재시도/백오프
  * X-User-Email 헤더 & body.user_email 동시 전송 (서버 어느 쪽을 봐도 호환)
  * 에러 발생 시 서버 응답 바디 일부 출력
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Any, Dict, Optional

import httpx
from httpx import ReadTimeout, ConnectTimeout, HTTPStatusError

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except Exception:
    # rich가 없다면 간단한 대체
    class _DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)
    Console = _DummyConsole  # type: ignore
    Table = None
    box = None

console = Console() if callable(getattr(Console, "__call__", None)) else Console


# ------------------------------
# 공통 HTTP 유틸
# ------------------------------
async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    backoff: float = 1.5,
) -> httpx.Response:
    """
    5xx/타임아웃에 대해 재시도. 5xx면 응답 바디 일부를 바로 출력해 원인 파악에 도움.
    """
    last_exc: Optional[Exception] = None
    for i in range(retries):
        try:
            resp = await client.request(method.upper(), url, params=params, json=json_data)
            if resp.status_code >= 500:
                # 서버 에러 바디 로그
                body = ""
                try:
                    body = resp.text
                except Exception:
                    body = "<no body>"
                snippet = body[:800]
                console.print(f"[red]SERVER 5xx on {url}[/red]\n{snippet}")
                raise HTTPStatusError("server error", request=resp.request, response=resp)
            return resp
        except (ReadTimeout, ConnectTimeout, HTTPStatusError) as e:
            last_exc = e
            if i == retries - 1:
                raise
            await asyncio.sleep(backoff ** i)
    # 여기 오지 않음
    raise last_exc if last_exc else RuntimeError("unknown error")


# ------------------------------
# 출력 도우미
# ------------------------------
def h1(title: str):
    console.print(f"\n[bold cyan]{title}[/bold cyan]")

def pretty_json(data: Any):
    try:
        console.print_json(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        console.print(json.dumps(data, ensure_ascii=False, indent=2))

def print_table_kv(title: str, kv: Dict[str, Any]):
    if Table is None:
        console.print(f"\n{title}")
        for k, v in kv.items():
            console.print(f"- {k}: {v}")
        return
    table = Table(title=title, box=box.SIMPLE if box else None)
    table.add_column("키")
    table.add_column("값", overflow="fold")
    for k, v in kv.items():
        table.add_row(str(k), str(v))
    console.print(table)


# ------------------------------
# 각 데모 단계
# ------------------------------
async def step_health(client: httpx.AsyncClient):
    h1("1. 서버 헬스체크")
    r = await request_with_retry(client, "GET", "/api/v1/health/full")
    data = r.json()
    # 주요 정보만 하이라이트
    services = data.get("services", {})
    llm = services.get("llm", {})
    rag = services.get("rag", {})
    gpu = services.get("gpu", {})
    summary = {
        "timestamp": data.get("timestamp"),
        "llm_status": llm.get("status"),
        "rag_status": rag.get("status"),
        "qdrant_points": rag.get("collections", {}).get("points_count"),
        "gpu_available": gpu.get("available"),
    }
    print_table_kv("헬스 요약", summary)


async def step_production(client: httpx.AsyncClient, query: str):
    h1("2. 생산 모니터링 시나리오")
    r = await request_with_retry(
        client, "POST", "/api/v1/scenarios/production",
        params={"query": query},
    )
    data = r.json()
    # 핵심만 출력
    report = data.get("report", "")
    total = (data.get("data") or {}).get("total", {})
    issues = data.get("issues", [])
    print_table_kv("생산 합계", {
        "계획": total.get("planned"),
        "실적": total.get("actual"),
        "달성률": total.get("achievement_rate"),
        "불량률": (total.get("defect_rate_pct") or total.get("defect_rate")),
    })
    console.print("\n[bold]주요 이슈[/bold]")
    pretty_json(issues)
    console.print("\n[bold]보고서[/bold]\n" + (report[:1200] + ("..." if len(report) > 1200 else "")))


async def step_maintenance(client: httpx.AsyncClient, equipment_id: str):
    h1("3. 예지보전 시나리오")
    r = await request_with_retry(
        client, "POST", "/api/v1/scenarios/maintenance",
        params={"equipment_id": equipment_id},
    )
    data = r.json()
    print_table_kv("예측 요약", {
        "설비": data.get("equipment_id"),
        "고장 확률": data.get("failure_probability"),
        "예상 정지시간": data.get("estimated_downtime"),
        "예상 시점": data.get("predicted_failure_time"),
    })
    console.print("\n[bold]AI 분석[/bold]\n" + str(data.get("ai_analysis", ""))[:1200])


async def step_quality(client: httpx.AsyncClient):
    h1("4. 품질 분석 시나리오")
    r = await request_with_retry(client, "POST", "/api/v1/scenarios/quality")
    data = r.json()
    summary = data.get("summary", {})
    print_table_kv("품질 요약", {
        "기간": data.get("period"),
        "불량률": summary.get("defect_rate"),
        "Cpk": summary.get("cpk"),
        "트렌드": summary.get("trend"),
    })
    console.print("\n[bold]주요 패턴[/bold]")
    pretty_json(data.get("patterns", []))
    console.print("\n[bold]근본 원인[/bold]")
    pretty_json(data.get("root_causes", []))
    console.print("\n[bold]AI 분석[/bold]\n" + str(data.get("ai_analysis", ""))[:1200])


async def step_chat(client: httpx.AsyncClient, message: str, user_email: str):
    h1("5. 대화형 AI 어시스턴트")
    console.print(f"\n질문: {message}")
    r = await request_with_retry(
        client, "POST", "/api/v1/chat",
        json_data={"message": message, "user_email": user_email},
    )
    data = r.json()
    console.print(f"[green]AI: {str(data.get('response', ''))[:1000]}")


# ------------------------------
# 메인 루틴
# ------------------------------
async def run_demo(args):
    base_url = args.base_url or os.getenv("KXN_API_BASE", "http://localhost:8000")
    user_email = args.user_email or os.getenv("KXN_DEMO_EMAIL", "demo@kainexa.local")

    timeout = httpx.Timeout(
        connect=float(os.getenv("KXN_HTTP_CONNECT_TIMEOUT", "30")),
        read=float(os.getenv("KXN_HTTP_READ_TIMEOUT", "120")),
        write=float(os.getenv("KXN_HTTP_WRITE_TIMEOUT", "60")),
        pool=float(os.getenv("KXN_HTTP_POOL_TIMEOUT", "60")),
    )

    headers = {
        "X-User-Email": user_email,        # 서버가 헤더를 우선 볼 때 대비
        "Accept": "application/json",
    }

    console.print(f"[bold]Base URL:[/bold] {base_url}")
    console.print(f"[bold]Demo User:[/bold] {user_email}")

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=headers) as client:
        # 1) 서버 헬스
        await step_health(client)

        # 2) 생산
        await step_production(client, query="어제 밤사 생산 현황 보고해줘")

        # 3) 예지보전
        await step_maintenance(client, equipment_id=args.equipment_id or "CNC_007")

        # 4) 품질
        await step_quality(client)

        # 5) 챗
        await step_chat(client, message="스마트 팩토리에서 OEE를 개선하는 방법은?", user_email=user_email)


def parse_args():
    p = argparse.ArgumentParser(description="Kainexa AI 데모 실행기")
    p.add_argument("--base-url", default=None, help="API 베이스 URL (기본: http://localhost:8000)")
    p.add_argument("--user-email", default=None, help="데모 사용자 이메일 (기본: demo@kainexa.local)")
    p.add_argument("--equipment-id", default=None, help="예지보전 설비 ID (기본: CNC_007)")
    return p.parse_args()


def main():
    try:
        args = parse_args()
        asyncio.run(run_demo(args))
    except KeyboardInterrupt:
        console.print("\n[red]중단됨[/red]")
    except Exception as e:
        console.print(f"[red]에러:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
