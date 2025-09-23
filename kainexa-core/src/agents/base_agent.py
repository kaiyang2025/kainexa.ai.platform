# ============================================
# src/agents/base_agent.py - 베이스 에이전트
# ============================================
"""src/agents/base_agent.py"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import uuid
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """에이전트 베이스 클래스"""
    
    def __init__(self, agent_type: str):
        self.agent_id = str(uuid.uuid4())
        self.agent_type = agent_type
        self.created_at = datetime.now()
        self.state = {}
        
        logger.info(f"Agent created: {agent_type} ({self.agent_id})")
    
    @abstractmethod
    async def process(self, **kwargs) -> Dict[str, Any]:
        """메인 처리 메서드"""
        pass
    
    async def initialize(self):
        """초기화"""
        logger.info(f"Initializing agent: {self.agent_id}")
    
    async def cleanup(self):
        """정리"""
        logger.info(f"Cleaning up agent: {self.agent_id}")
    
    def get_state(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'created_at': self.created_at.isoformat(),
            'state': self.state
        }
    
    def update_state(self, key: str, value: Any):
        """상태 업데이트"""
        self.state[key] = value