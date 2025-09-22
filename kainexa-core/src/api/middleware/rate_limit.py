# src/api/middleware/rate_limit.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
from src.core.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.rate_limit_records = defaultdict(list)
        self.max_requests = settings.RATE_LIMIT_REQUESTS
        self.window_seconds = settings.RATE_LIMIT_PERIOD
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        
        # Get current time
        current_time = time.time()
        
        # Clean old records
        self.rate_limit_records[client_ip] = [
            timestamp for timestamp in self.rate_limit_records[client_ip]
            if current_time - timestamp < self.window_seconds
        ]
        
        # Check rate limit
        if len(self.rate_limit_records[client_ip]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
        
        # Record request
        self.rate_limit_records[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        return response