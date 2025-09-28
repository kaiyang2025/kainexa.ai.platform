# ============================================
# src/core/cache.py - 통합 캐시 관리
# ============================================
"""src/core/cache.py"""
import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional
from datetime import timedelta

from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CacheManager:
    """통합 캐시 관리자 - Redis + 메모리 캐시"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}  # 빠른 접근용 메모리 캐시
        self.connect()
    
    def connect(self):
        """Redis 연결"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Redis connected")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def get(self, key: str, use_memory: bool = True) -> Optional[Any]:
        """캐시 조회"""
        # 1. 메모리 캐시 확인
        if use_memory and key in self.memory_cache:
            return self.memory_cache[key]
        
        # 2. Redis 캐시 확인
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value:
                    decoded = json.loads(value)
                    # 메모리 캐시 업데이트
                    if use_memory:
                        self.memory_cache[key] = decoded
                    return decoded
            except Exception as e:
                logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, 
                  ttl: int = 3600, use_memory: bool = True):
        """캐시 저장"""
        # 1. 메모리 캐시 저장
        if use_memory:
            self.memory_cache[key] = value
        
        # 2. Redis 저장
        if self.redis_client:
            try:
                serialized = json.dumps(value)
                await self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.error(f"Cache set error: {e}")
    
    async def delete(self, key: str):
        """캐시 삭제"""
        # 메모리 캐시 삭제
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Redis 삭제
        if self.redis_client:
            await self.redis_client.delete(key)
    
    async def clear(self):
        """전체 캐시 클리어"""
        self.memory_cache.clear()
        if self.redis_client:
            await self.redis_client.flushdb()