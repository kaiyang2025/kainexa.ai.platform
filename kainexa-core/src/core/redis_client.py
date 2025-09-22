# src/core/redis_client.py
import redis.asyncio as redis
from src.core.config import settings
import json
from typing import Optional, Any

class RedisClient:
    def __init__(self):
        self.redis = None
    
    async def connect(self):
        self.redis = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def disconnect(self):
        if self.redis:
            await self.redis.close()
    
    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        value_str = json.dumps(value)
        if ttl:
            await self.redis.setex(key, ttl, value_str)
        else:
            await self.redis.set(key, value_str)
    
    async def delete(self, key: str):
        await self.redis.delete(key)

redis_client = RedisClient()