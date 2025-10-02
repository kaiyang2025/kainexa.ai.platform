# src/core/policy/policy_engine.py
"""
Policy Engine for Workflow Execution
워크플로우 실행 중 정책(SLA, 폴백, 에스컬레이션, 비용 등)을 관리하고 적용
"""
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import structlog

from src.core.orchestration.execution_context import ExecutionContext, ExecutionStatus

logger = structlog.get_logger()

# ========== Policy Types ==========
class PolicyType(str, Enum):
    """정책 타입"""
    SLA = "sla"
    FALLBACK = "fallback"
    ESCALATION = "escalation"
    COST = "cost"
    SECURITY = "security"
    RATE_LIMIT = "rate_limit"
    CIRCUIT_BREAKER = "circuit_breaker"

class PolicyAction(str, Enum):
    """정책 액션"""
    CONTINUE = "continue"
    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"
    SWITCH_MODEL = "switch_model"
    USE_CACHE = "use_cache"
    TRANSFER_AGENT = "transfer_agent"
    NOTIFY = "notify"
    LOG = "log"

@dataclass
class PolicyViolation:
    """정책 위반"""
    policy_type: PolicyType
    policy_name: str
    violation: str
    severity: str  # low, medium, high, critical
    action: PolicyAction
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class PolicyResult:
    """정책 평가 결과"""
    passed: bool
    violations: List[PolicyViolation] = field(default_factory=list)
    actions: List[PolicyAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ========== Policy Handlers ==========
class PolicyHandler:
    """정책 핸들러 베이스"""
    
    async def evaluate(self, 
                       context: ExecutionContext,
                       policy_config: Dict[str, Any]) -> PolicyResult:
        """정책 평가"""
        raise NotImplementedError
    
    async def apply_action(self,
                          action: PolicyAction,
                          context: ExecutionContext,
                          metadata: Dict[str, Any]) -> bool:
        """액션 적용"""
        raise NotImplementedError

class SLAPolicyHandler(PolicyHandler):
    """SLA 정책 핸들러"""
    
    async def evaluate(self,
                      context: ExecutionContext,
                      policy_config: Dict[str, Any]) -> PolicyResult:
        
        result = PolicyResult(passed=True)
        
        # 최대 지연시간 체크
        max_latency_ms = policy_config.get('max_latency_ms')
        if max_latency_ms and context.started_at:
            elapsed_ms = int(
                (datetime.utcnow() - context.started_at).total_seconds() * 1000
            )
            
            if elapsed_ms > max_latency_ms:
                violation = PolicyViolation(
                    policy_type=PolicyType.SLA,
                    policy_name="max_latency",
                    violation=f"Latency {elapsed_ms}ms exceeds limit {max_latency_ms}ms",
                    severity="high",
                    action=PolicyAction.ABORT,
                    metadata={'elapsed_ms': elapsed_ms, 'limit_ms': max_latency_ms}
                )
                result.passe