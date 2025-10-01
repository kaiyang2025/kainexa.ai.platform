# src/api/middleware/rate_limit.py
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    아주 단순한 인메모리 레이트리미터
    - 키: 기본은 client IP (X-Forwarded-For > client.host 순)
    - 윈도우: window_seconds 동안 max_requests 개 허용
    - 응답 헤더:
        X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
      초과 시 429 + Retry-After
    """

    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: int = 60,
        identify: Optional[Callable[[Request], str]] = None,
    ) -> None:
        super().__init__(app)
        self.max_requests = int(max_requests)
        self.window = int(window_seconds)
        self._identify = identify or self._default_identify

        # 키별 요청 타임스탬프 큐
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        # 키별 락
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def dispatch(self, request: Request, call_next) -> Response:
        key = self._identify(request)
        now = time.monotonic()
        reset_at: float

        # 키 전용 락으로 경쟁 최소화
        lock = self._locks[key]
        async with lock:
            q = self._buckets[key]

            # 윈도우 밖(만료) 제거
            cutoff = now - self.window
            while q and q[0] <= cutoff:
                q.popleft()

            # 허용 여부 판단
            if len(q) >= self.max_requests:
                # 다음 요청 가능 시각(큐의 가장 오래된 + window)
                reset_at = q[0] + self.window
                retry_after = max(0, int(reset_at - now))

                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(self.max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(reset_at)),
                    },
                )

            # 허용 → 현재 타임스탬프 추가
            q.append(now)
            remaining = max(0, self.max_requests - len(q))
            reset_at = q[0] + self.window if q else now + self.window

        # 다운스트림 호출
        resp: Response = await call_next(request)
        # 헤더 부착
        resp.headers.setdefault("X-RateLimit-Limit", str(self.max_requests))
        resp.headers.setdefault("X-RateLimit-Remaining", str(remaining))
        resp.headers.setdefault("X-RateLimit-Reset", str(int(reset_at)))
        return resp

    @staticmethod
    def _default_identify(request: Request) -> str:
        # 프록시 뒤라면 X-Forwarded-For 우선
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # "client, proxy1, proxy2" → 첫 요소
            return xff.split(",")[0].strip()
        client_host = getattr(request.client, "host", None)
        return client_host or "anonymous"
