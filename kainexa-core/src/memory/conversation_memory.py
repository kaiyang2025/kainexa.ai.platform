# kainexa-core/src/memory/conversation_memory.py
"""
대화 메모리 시스템 - 단기/장기 메모리 관리
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import asyncio
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """메모리 타입"""
    SHORT_TERM = "short_term"  # 현재 세션
    WORKING = "working"        # 작업 메모리 (1일)
    LONG_TERM = "long_term"    # 장기 메모리 (영구)


@dataclass
class MemoryItem:
    """메모리 항목"""
    id: str
    user_id: str
    session_id: str
    type: MemoryType
    content: str
    role: str  # user/assistant/system
    metadata: Dict[str, Any]
    timestamp: datetime
    importance: float = 0.5  # 0-1 중요도
    accessed_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['type'] = self.type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.last_accessed:
            data['last_accessed'] = self.last_accessed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryItem':
        data['type'] = MemoryType(data['type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_accessed'):
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class ConversationMemory:
    """대화 메모리 관리 시스템"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_short_term: int = 20,      # 단기 메모리 최대 개수
        max_working: int = 100,        # 작업 메모리 최대 개수
        working_ttl_hours: int = 24,   # 작업 메모리 유효시간
    ):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.max_short_term = max_short_term
        self.max_working = max_working
        self.working_ttl = timedelta(hours=working_ttl_hours)
        
        # 메모리 버퍼 (빠른 액세스용)
        self.short_term_buffer: Dict[str, List[MemoryItem]] = {}
    
    async def connect(self):
        """Redis 연결"""
        self.redis_client = redis.Redis.from_url(
            self.redis_url,
            decode_responses=True
        )
        await self.redis_client.ping()
        logger.info("Memory system connected to Redis")
    
    async def disconnect(self):
        """연결 종료"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _generate_id(self, content: str, user_id: str) -> str:
        """메모리 ID 생성"""
        data = f"{user_id}:{content}:{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    # ========== 메모리 저장 ==========
    
    async def remember(
        self,
        user_id: str,
        session_id: str,
        content: str,
        role: str = "user",
        metadata: Optional[Dict] = None,
        importance: float = 0.5
    ) -> MemoryItem:
        """메모리 저장"""
        
        # 메모리 아이템 생성
        memory = MemoryItem(
            id=self._generate_id(content, user_id),
            user_id=user_id,
            session_id=session_id,
            type=MemoryType.SHORT_TERM,
            content=content,
            role=role,
            metadata=metadata or {},
            timestamp=datetime.now(),
            importance=importance
        )
        
        # 1. 단기 메모리에 저장
        await self._save_short_term(memory)
        
        # 2. 중요도가 높으면 작업 메모리에도 저장
        if importance > 0.7:
            memory.type = MemoryType.WORKING
            await self._save_working(memory)
        
        # 3. 매우 중요하면 장기 메모리에도 저장
        if importance > 0.9:
            memory.type = MemoryType.LONG_TERM
            await self._save_long_term(memory)
        
        logger.info(f"Memory saved: {memory.id} (importance: {importance})")
        return memory
    
    async def _save_short_term(self, memory: MemoryItem):
        """단기 메모리 저장 (현재 세션)"""
        key = f"memory:short:{memory.user_id}:{memory.session_id}"
        
        # 리스트에 추가
        await self.redis_client.lpush(key, json.dumps(memory.to_dict()))
        
        # 최대 개수 유지
        await self.redis_client.ltrim(key, 0, self.max_short_term - 1)
        
        # TTL 설정 (3시간)
        await self.redis_client.expire(key, 10800)
        
        # 버퍼 업데이트
        buffer_key = f"{memory.user_id}:{memory.session_id}"
        if buffer_key not in self.short_term_buffer:
            self.short_term_buffer[buffer_key] = []
        self.short_term_buffer[buffer_key].insert(0, memory)
        self.short_term_buffer[buffer_key] = self.short_term_buffer[buffer_key][:self.max_short_term]
    
    async def _save_working(self, memory: MemoryItem):
        """작업 메모리 저장 (1일 유지)"""
        key = f"memory:working:{memory.user_id}"
        
        # Sorted Set으로 저장 (importance를 score로)
        await self.redis_client.zadd(
            key,
            {json.dumps(memory.to_dict()): memory.importance}
        )
        
        # 오래된 항목 제거
        await self.redis_client.zremrangebyrank(key, 0, -self.max_working-1)
        
        # TTL 갱신
        await self.redis_client.expire(key, int(self.working_ttl.total_seconds()))
    
    async def _save_long_term(self, memory: MemoryItem):
        """장기 메모리 저장 (영구)"""
        key = f"memory:long:{memory.user_id}"
        
        # Hash로 저장
        await self.redis_client.hset(
            key,
            memory.id,
            json.dumps(memory.to_dict())
        )
        
        # 인덱스 업데이트 (검색용)
        index_key = f"memory:index:{memory.user_id}"
        await self.redis_client.zadd(
            index_key,
            {memory.id: memory.timestamp.timestamp()}
        )
    
    # ========== 메모리 조회 ==========
    
    async def recall(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[MemoryItem]:
        """메모리 조회"""
        
        memories = []
        
        if memory_type == MemoryType.SHORT_TERM:
            memories = await self._get_short_term(user_id, session_id, limit)
        
        elif memory_type == MemoryType.WORKING:
            memories = await self._get_working(user_id, limit, min_importance)
        
        elif memory_type == MemoryType.LONG_TERM:
            memories = await self._get_long_term(user_id, limit)
        
        # 접근 횟수 업데이트
        for memory in memories:
            memory.accessed_count += 1
            memory.last_accessed = datetime.now()
        
        return memories
    
    async def _get_short_term(
        self,
        user_id: str,
        session_id: str,
        limit: int
    ) -> List[MemoryItem]:
        """단기 메모리 조회"""
        
        # 버퍼 확인
        buffer_key = f"{user_id}:{session_id}"
        if buffer_key in self.short_term_buffer:
            return self.short_term_buffer[buffer_key][:limit]
        
        # Redis에서 조회
        key = f"memory:short:{user_id}:{session_id}"
        items = await self.redis_client.lrange(key, 0, limit - 1)
        
        memories = []
        for item in items:
            data = json.loads(item)
            memories.append(MemoryItem.from_dict(data))
        
        return memories
    
    async def _get_working(
        self,
        user_id: str,
        limit: int,
        min_importance: float
    ) -> List[MemoryItem]:
        """작업 메모리 조회"""
        key = f"memory:working:{user_id}"
        
        # 중요도 기준으로 조회
        items = await self.redis_client.zrevrangebyscore(
            key,
            max=1.0,
            min=min_importance,
            start=0,
            num=limit
        )
        
        memories = []
        for item in items:
            data = json.loads(item)
            memories.append(MemoryItem.from_dict(data))
        
        return memories
    
    async def _get_long_term(
        self,
        user_id: str,
        limit: int
    ) -> List[MemoryItem]:
        """장기 메모리 조회"""
        
        # 최근 항목 조회
        index_key = f"memory:index:{user_id}"
        memory_ids = await self.redis_client.zrevrange(index_key, 0, limit - 1)
        
        if not memory_ids:
            return []
        
        # 메모리 내용 조회
        key = f"memory:long:{user_id}"
        items = await self.redis_client.hmget(key, memory_ids)
        
        memories = []
        for item in items:
            if item:
                data = json.loads(item)
                memories.append(MemoryItem.from_dict(data))
        
        return memories
    
    # ========== 메모리 요약 ==========
    
    async def summarize_session(
        self,
        user_id: str,
        session_id: str,
        llm_func=None
    ) -> str:
        """세션 대화 요약"""
        
        # 단기 메모리 조회
        memories = await self.recall(
            user_id,
            session_id,
            MemoryType.SHORT_TERM,
            limit=50
        )
        
        if not memories:
            return ""
        
        # 대화 내용 구성
        conversation = []
        for mem in reversed(memories):
            conversation.append(f"{mem.role}: {mem.content}")
        
        # LLM으로 요약 생성
        if llm_func:
            prompt = f"""다음 대화를 간결하게 요약하세요:
            
{chr(10).join(conversation)}

요약:"""
            summary = await llm_func(prompt)
        else:
            # 간단한 요약
            summary = f"총 {len(memories)}개 메시지 교환. "
            topics = set()
            for mem in memories:
                if mem.metadata.get('intent'):
                    topics.add(mem.metadata['intent'])
            if topics:
                summary += f"주요 주제: {', '.join(topics)}"
        
        # 요약을 작업 메모리에 저장
        await self.remember(
            user_id,
            session_id,
            summary,
            role="system",
            metadata={"type": "summary", "message_count": len(memories)},
            importance=0.8
        )
        
        return summary
    
    # ========== 메모리 관리 ==========
    
    async def forget(
        self,
        user_id: str,
        memory_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None
    ):
        """메모리 삭제"""
        
        if memory_id and memory_type == MemoryType.LONG_TERM:
            # 특정 장기 메모리 삭제
            key = f"memory:long:{user_id}"
            await self.redis_client.hdel(key, memory_id)
            
            index_key = f"memory:index:{user_id}"
            await self.redis_client.zrem(index_key, memory_id)
        
        elif memory_type == MemoryType.SHORT_TERM:
            # 단기 메모리 전체 삭제
            pattern = f"memory:short:{user_id}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
            
            # 버퍼 클리어
            self.short_term_buffer = {
                k: v for k, v in self.short_term_buffer.items()
                if not k.startswith(user_id)
            }
        
        logger.info(f"Memory forgotten: user={user_id}, type={memory_type}")
    
    async def optimize(self):
        """메모리 최적화 (오래된 항목 정리)"""
        
        # 1개월 이상 접근하지 않은 작업 메모리 삭제
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # 모든 사용자의 작업 메모리 검토
        pattern = "memory:working:*"
        keys = await self.redis_client.keys(pattern)
        
        for key in keys:
            items = await self.redis_client.zrange(key, 0, -1)
            for item in items:
                data = json.loads(item)
                memory = MemoryItem.from_dict(data)
                
                if memory.last_accessed and memory.last_accessed < cutoff_date:
                    await self.redis_client.zrem(key, item)
                    logger.info(f"Optimized: removed old memory {memory.id}")
    
    # ========== 컨텍스트 구성 ==========
    
    async def build_context(
        self,
        user_id: str,
        session_id: str,
        query: str = "",
        max_tokens: int = 2000
    ) -> str:
        """대화 컨텍스트 구성"""
        
        context_parts = []
        token_count = 0
        
        # 1. 단기 메모리 (현재 대화)
        short_term = await self.recall(
            user_id, session_id, 
            MemoryType.SHORT_TERM, limit=10
        )
        
        if short_term:
            recent_conv = []
            for mem in reversed(short_term[-5:]):  # 최근 5개
                recent_conv.append(f"{mem.role}: {mem.content}")
            context_parts.append("최근 대화:\n" + "\n".join(recent_conv))
            token_count += len(" ".join(recent_conv).split())
        
        # 2. 관련 작업 메모리
        if token_count < max_tokens * 0.7:
            working = await self.recall(
                user_id, None,
                MemoryType.WORKING, limit=5,
                min_importance=0.7
            )
            
            if working:
                important_info = []
                for mem in working:
                    if query and query.lower() in mem.content.lower():
                        important_info.append(mem.content)
                
                if important_info:
                    context_parts.append("관련 정보:\n" + "\n".join(important_info))
        
        # 3. 사용자 프로필 (장기 메모리에서)
        if token_count < max_tokens * 0.9:
            profile = await self._get_user_profile(user_id)
            if profile:
                context_parts.append(f"사용자 정보: {profile}")
        
        return "\n\n".join(context_parts)
    
    async def _get_user_profile(self, user_id: str) -> str:
        """사용자 프로필 조회"""
        
        # 장기 메모리에서 프로필 정보 조회
        long_term = await self.recall(
            user_id, None,
            MemoryType.LONG_TERM, limit=10
        )
        
        profile_items = []
        for mem in long_term:
            if mem.metadata.get('type') == 'profile':
                profile_items.append(mem.content)
        
        return ". ".join(profile_items) if profile_items else ""


# ========== 사용 예제 ==========

async def example_usage():
    """Memory 시스템 사용 예제"""
    
    memory = ConversationMemory()
    await memory.connect()
    
    user_id = "user123"
    session_id = "session456"
    
    # 대화 저장
    await memory.remember(
        user_id, session_id,
        "안녕하세요. 제품 주문 상태를 확인하고 싶습니다.",
        role="user",
        metadata={"intent": "order_status"},
        importance=0.6
    )
    
    await memory.remember(
        user_id, session_id,
        "안녕하세요! 주문번호를 알려주시면 확인해드리겠습니다.",
        role="assistant",
        importance=0.5
    )
    
    # 중요 정보 저장 (작업 메모리로)
    await memory.remember(
        user_id, session_id,
        "주문번호는 ORD-2024-1234입니다",
        role="user",
        metadata={"order_id": "ORD-2024-1234"},
        importance=0.8
    )
    
    # 컨텍스트 구성
    context = await memory.build_context(user_id, session_id, "주문")
    print(f"Context:\n{context}")
    
    # 세션 요약
    summary = await memory.summarize_session(user_id, session_id)
    print(f"Summary: {summary}")
    
    await memory.disconnect()


if __name__ == "__main__":
    asyncio.run(example_usage())