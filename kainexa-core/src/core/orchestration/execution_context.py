# src/core/orchestration/execution_context.py
"""
Workflow Execution Context
워크플로우 실행 중 상태와 변수를 관리
"""
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
import json
import copy
from enum import Enum

class ExecutionStatus(str, Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class NodeStatus(str, Enum):
    """노드 실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class NodeResult:
    """노드 실행 결과"""
    node_id: str
    status: NodeStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    """실행 메트릭"""
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    skipped_nodes: int = 0
    total_duration_ms: int = 0
    tokens_used: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    model_inference_ms: int = 0

class ExecutionContext:
    """
    워크플로우 실행 컨텍스트
    실행 중 모든 상태와 데이터를 관리
    """
    
    def __init__(self,
                 execution_id: Optional[str] = None,
                 workflow_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 tenant_id: Optional[str] = None,
                 environment: str = "prod",
                 channel: str = "api",
                 language: str = "ko"):
        
        # 식별자
        self.execution_id = execution_id or str(uuid4())
        self.workflow_id = workflow_id
        self.session_id = session_id
        self.user_id = user_id
        self.tenant_id = tenant_id
        
        # 환경 정보
        self.environment = environment
        self.channel = channel
        self.language = language
        
        # 실행 상태
        self.status = ExecutionStatus.PENDING
        self.started_at = None
        self.completed_at = None
        
        # 변수 저장소
        self.variables: Dict[str, Any] = {}
        self.global_variables: Dict[str, Any] = {}
        
        # 노드 실행 결과
        self.node_results: Dict[str, NodeResult] = {}
        self.execution_path: List[str] = []
        
        # 메트릭
        self.metrics = ExecutionMetrics()
        
        # 에러 추적
        self.errors: List[Dict[str, Any]] = []
        
        # 실행 이력
        self.history: List[Dict[str, Any]] = []
        
        # 캐시
        self.cache: Dict[str, Any] = {}
        
        # 정책
        self.policies: Dict[str, Any] = {}
        
        # 중단 플래그
        self.should_stop = False
        self.stop_reason = None
    
    # ========== 변수 관리 ==========
    def get_variable(self, key: str, default: Any = None) -> Any:
        """변수 조회"""
        # 먼저 로컬 변수 확인
        if key in self.variables:
            return self.variables[key]
        # 글로벌 변수 확인
        if key in self.global_variables:
            return self.global_variables[key]
        return default
    
    def set_variable(self, key: str, value: Any, is_global: bool = False):
        """변수 설정"""
        if is_global:
            self.global_variables[key] = value
        else:
            self.variables[key] = value
        
        # 히스토리 기록
        self.add_history({
            'action': 'set_variable',
            'key': key,
            'value': value,
            'is_global': is_global,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def update_variables(self, variables: Dict[str, Any], is_global: bool = False):
        """여러 변수 일괄 업데이트"""
        if is_global:
            self.global_variables.update(variables)
        else:
            self.variables.update(variables)
    
    def get_all_variables(self) -> Dict[str, Any]:
        """모든 변수 반환 (글로벌 + 로컬)"""
        all_vars = {}
        all_vars.update(self.global_variables)
        all_vars.update(self.variables)
        return all_vars
    
    # ========== 노드 실행 관리 ==========
    def start_node_execution(self, node_id: str):
        """노드 실행 시작"""
        self.execution_path.append(node_id)
        
        result = NodeResult(
            node_id=node_id,
            status=NodeStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        self.node_results[node_id] = result
        self.metrics.total_nodes += 1
        
        self.add_history({
            'action': 'node_start',
            'node_id': node_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def complete_node_execution(self, 
                               node_id: str,
                               outputs: Dict[str, Any],
                               metrics: Optional[Dict[str, Any]] = None):
        """노드 실행 완료"""
        if node_id not in self.node_results:
            raise ValueError(f"Node {node_id} was not started")
        
        result = self.node_results[node_id]
        result.status = NodeStatus.SUCCESS
        result.outputs = outputs
        result.completed_at = datetime.utcnow()
        result.duration_ms = int(
            (result.completed_at - result.started_at).total_seconds() * 1000
        )
        
        if metrics:
            result.metrics = metrics
            # 메트릭 집계
            self.aggregate_metrics(metrics)
        
        self.metrics.completed_nodes += 1
        
        # 변수 업데이트 (노드 출력을 변수로 저장)
        for key, value in outputs.items():
            self.set_variable(f"{node_id}.{key}", value)
        
        self.add_history({
            'action': 'node_complete',
            'node_id': node_id,
            'duration_ms': result.duration_ms,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def fail_node_execution(self, node_id: str, error: str):
        """노드 실행 실패"""
        if node_id not in self.node_results:
            raise ValueError(f"Node {node_id} was not started")
        
        result = self.node_results[node_id]
        result.status = NodeStatus.FAILED
        result.error = error
        result.completed_at = datetime.utcnow()
        result.duration_ms = int(
            (result.completed_at - result.started_at).total_seconds() * 1000
        )
        
        self.metrics.failed_nodes += 1
        
        # 에러 기록
        self.errors.append({
            'node_id': node_id,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        self.add_history({
            'action': 'node_fail',
            'node_id': node_id,
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def skip_node_execution(self, node_id: str, reason: str = ""):
        """노드 실행 스킵"""
        result = NodeResult(
            node_id=node_id,
            status=NodeStatus.SKIPPED,
            outputs={'reason': reason}
        )
        
        self.node_results[node_id] = result
        self.metrics.skipped_nodes += 1
        
        self.add_history({
            'action': 'node_skip',
            'node_id': node_id,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def get_node_result(self, node_id: str) -> Optional[NodeResult]:
        """노드 실행 결과 조회"""
        return self.node_results.get(node_id)
    
    def get_node_output(self, node_id: str, key: str = None, default: Any = None) -> Any:
        """노드 출력 값 조회"""
        result = self.node_results.get(node_id)
        if not result:
            return default
        
        if key:
            return result.outputs.get(key, default)
        return result.outputs
    
    # ========== 실행 상태 관리 ==========
    def start_execution(self):
        """워크플로우 실행 시작"""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        
        self.add_history({
            'action': 'execution_start',
            'timestamp': self.started_at.isoformat()
        })
    
    def complete_execution(self):
        """워크플로우 실행 완료"""
        self.status = ExecutionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.metrics.total_duration_ms = int(
            (self.completed_at - self.started_at).total_seconds() * 1000
        )
        
        self.add_history({
            'action': 'execution_complete',
            'duration_ms': self.metrics.total_duration_ms,
            'timestamp': self.completed_at.isoformat()
        })
    
    def fail_execution(self, error: str):
        """워크플로우 실행 실패"""
        self.status = ExecutionStatus.FAILED
        self.completed_at = datetime.utcnow()
        
        if self.started_at:
            self.metrics.total_duration_ms = int(
                (self.completed_at - self.started_at).total_seconds() * 1000
            )
        
        self.errors.append({
            'error': error,
            'timestamp': self.completed_at.isoformat()
        })
        
        self.add_history({
            'action': 'execution_fail',
            'error': error,
            'timestamp': self.completed_at.isoformat()
        })
    
    def cancel_execution(self, reason: str = ""):
        """워크플로우 실행 취소"""
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        self.should_stop = True
        self.stop_reason = reason
        
        self.add_history({
            'action': 'execution_cancel',
            'reason': reason,
            'timestamp': self.completed_at.isoformat()
        })
    
    # ========== 캐시 관리 ==========
    def get_cache(self, key: str, default: Any = None) -> Any:
        """캐시 조회"""
        return self.cache.get(key, default)
    
    def set_cache(self, key: str, value: Any):
        """캐시 설정"""
        self.cache[key] = value
        self.metrics.cache_hits += 1
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache.clear()
    
    # ========== 메트릭 관리 ==========
    def aggregate_metrics(self, metrics: Dict[str, Any]):
        """메트릭 집계"""
        if 'tokens' in metrics:
            self.metrics.tokens_used += metrics['tokens']
        if 'api_calls' in metrics:
            self.metrics.api_calls += metrics['api_calls']
        if 'inference_ms' in metrics:
            self.metrics.model_inference_ms += metrics['inference_ms']
    
    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 반환"""
        return {
            'total_nodes': self.metrics.total_nodes,
            'completed_nodes': self.metrics.completed_nodes,
            'failed_nodes': self.metrics.failed_nodes,
            'skipped_nodes': self.metrics.skipped_nodes,
            'total_duration_ms': self.metrics.total_duration_ms,
            'tokens_used': self.metrics.tokens_used,
            'api_calls': self.metrics.api_calls,
            'cache_hits': self.metrics.cache_hits,
            'model_inference_ms': self.metrics.model_inference_ms
        }
    
    # ========== 히스토리 관리 ==========
    def add_history(self, entry: Dict[str, Any]):
        """히스토리 추가"""
        self.history.append(entry)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """히스토리 조회"""
        return self.history
    
    # ========== 정책 관리 ==========
    def set_policies(self, policies: Dict[str, Any]):
        """정책 설정"""
        self.policies = policies
    
    def get_policy(self, key: str, default: Any = None) -> Any:
        """정책 조회"""
        # 중첩된 키 지원 (예: "sla.max_latency_ms")
        keys = key.split('.')
        value = self.policies
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    # ========== 유틸리티 ==========
    def clone(self) -> 'ExecutionContext':
        """컨텍스트 복제 (병렬 실행용)"""
        cloned = ExecutionContext(
            execution_id=f"{self.execution_id}-clone",
            workflow_id=self.workflow_id,
            session_id=self.session_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            environment=self.environment,
            channel=self.channel,
            language=self.language
        )
        
        # 변수 복사 (deep copy)
        cloned.variables = copy.deepcopy(self.variables)
        cloned.global_variables = self.global_variables  # 글로벌은 참조 공유
        cloned.policies = self.policies  # 정책도 참조 공유
        
        return cloned
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'execution_id': self.execution_id,
            'workflow_id': self.workflow_id,
            'session_id': self.session_id,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'variables': self.variables,
            'execution_path': self.execution_path,
            'metrics': self.get_metrics(),
            'errors': self.errors
        }
    
    def __repr__(self) -> str:
        return f"ExecutionContext(id={self.execution_id}, status={self.status})"