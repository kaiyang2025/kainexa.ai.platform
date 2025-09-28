# ============================================
# src/api/middleware/middleware.py - 통합 미들웨어
# ============================================
"""src/api/middleware/middleware.py"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import json
from typing import Callable
from collections import defaultdict
import asyncio

from src.utils.logger import get_logger

logger = get_logger(__name__)

class UnifiedMiddleware(BaseHTTPMiddleware):
    """통합 미들웨어 - Rate Limiting, Logging, Error Handling"""
    
    def __init__(self, app, 
                 rate_limit: int = 100,
                 window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.window = window
        self.request_counts = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 1. Request ID 생성
        request_id = f"{time.time()}-{hash(request.url.path)}"
        request.state.request_id = request_id
        
        # 2. Rate Limiting
        client_ip = request.client.host
        if not self._check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # 3. Request Logging
        start_time = time.time()
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")
        
        try:
            # 4. Process Request
            response = await call_next(request)
            
            # 5. Response Logging
            process_time = time.time() - start_time
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            logger.info(f"Response {request_id}: {response.status_code} in {process_time:.3f}s")
            
            return response
            
        except Exception as e:
            # 6. Error Handling
            logger.error(f"Error {request_id}: {str(e)}")
            return Response(
                content=json.dumps({
                    "error": str(e),
                    "request_id": request_id
                }),
                status_code=500,
                media_type="application/json"
            )
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Rate limiting 체크"""
        now = time.time()
        
        # 오래된 요청 제거
        self.request_counts[client_ip] = [
            timestamp for timestamp in self.request_counts[client_ip]
            if now - timestamp < self.window
        ]
        
        # 제한 확인
        if len(self.request_counts[client_ip]) >= self.rate_limit:
            return False
        
        # 요청 기록
        self.request_counts[client_ip].append(now)
        return True
