# src/core/executor/graph_executor.py
"""
Workflow Graph Executor
컴파일된 워크플로우 그래프를 실행하는 핵심 엔진
"""
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import re
from uuid import uuid4
import structlog

from src.core.executor.execution_context import (
    ExecutionContext, 
    ExecutionStatus,
    NodeStatus,
    NodeResult
)
from src.core.executor.node_executors import get_node_executor

logger = structlog.get_logger()

class ExecutionResult:
    """워크플로우 실행 결과"""
    
    def __init__(self,
                 execution_id: str,
                 status: ExecutionStatus,
                 outputs: Dict[str, Any],
                 metrics: Dict[str, Any],
                 errors: List[Dict[str, Any]] = None):
        self.execution_id = execution_id
        self.status = status
        self.outputs = outputs
        self.metrics = metrics
        self.errors = errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'status': self.status.value,
            'outputs': self.outputs,
            'metrics': self.metrics,
            'errors': self.errors
        }

class GraphExecutor:
    """
    워크플로우 그래프 실행기
    DAG(Directed Acyclic Graph) 형태의 워크플로우를 실행
    """
    
    def __init__(self):
        self.running_executions: Dict[str, ExecutionContext] = {}
        
    async def execute(self,
                     graph: Dict[str, Any],
                     input_data: Dict[str, Any],
                     context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """
        워크플로우 그래프 실행
        
        Args:
            graph: 컴파일된 그래프 구조
            input_data: 입력 데이터
            context: 실행 컨텍스트 (선택적)
            
        Returns:
            ExecutionResult: 실행 결과
        """
        
        # 컨텍스트 생성 또는 사용
        if context is None:
            context = ExecutionContext()
        
        # 워크플로우 ID 설정
        context.workflow_id = graph.get('metadata', {}).get('namespace', '') + '/' + \
                            graph.get('metadata', {}).get('name', 'unknown')
        
        # 정책 설정
        context.set_policies(graph.get('policies', {}))
        
        # 입력 데이터를 변수로 설정
        context.update_variables(input_data)
        
        # 실행 시작
        context.start_execution()
        self.running_executions[context.execution_id] = context
        
        logger.info(
            "Starting workflow execution",
            execution_id=context.execution_id,
            workflow_id=context.workflow_id
        )
        
        try:
            # 타임아웃 설정
            timeout_ms = context.get_policy('sla.timeout_ms', 30000)
            timeout = timeout_ms / 1000.0
            
            # 그래프 실행 (타임아웃 적용)
            await asyncio.wait_for(
                self._execute_graph(graph, context),
                timeout=timeout
            )
            
            # 실행 완료
            context.complete_execution()
            
        except asyncio.TimeoutError:
            logger.error(
                "Workflow execution timeout",
                execution_id=context.execution_id,
                timeout_ms=timeout_ms
            )
            context.status = ExecutionStatus.TIMEOUT
            context.fail_execution(f"Execution timeout after {timeout_ms}ms")
            
        except Exception as e:
            logger.error(
                "Workflow execution failed",
                execution_id=context.execution_id,
                error=str(e)
            )
            context.fail_execution(str(e))
            
        finally:
            # 실행 목록에서 제거
            self.running_executions.pop(context.execution_id, None)
        
        # 결과 생성
        return ExecutionResult(
            execution_id=context.execution_id,
            status=context.status,
            outputs=self._extract_outputs(context),
            metrics=context.get_metrics(),
            errors=context.errors
        )
    
    async def _execute_graph(self,
                            graph: Dict[str, Any],
                            context: ExecutionContext):
        """그래프 실행 로직"""
        
        nodes = graph.get('nodes', {})
        edges = graph.get('edges', {})
        entry_points = graph.get('entry_points', [])
        
        if not entry_points:
            raise ValueError("No entry points found in graph")
        
        # 노드 실행 상태 추적
        executed_nodes: Set[str] = set()
        pending_nodes: Set[str] = set(entry_points)
        
        # BFS 방식으로 그래프 실행
        while pending_nodes and not context.should_stop:
            # 실행 가능한 노드 찾기
            executable_nodes = self._find_executable_nodes(
                pending_nodes,
                executed_nodes,
                nodes,
                edges
            )
            
            if not executable_nodes:
                # 데드락 감지
                if pending_nodes:
                    logger.error(
                        "Deadlock detected",
                        pending=list(pending_nodes),
                        executed=list(executed_nodes)
                    )
                    raise RuntimeError("Deadlock detected in workflow")
                break
            
            # 병렬 실행 가능한 노드들 동시 실행
            tasks = []
            for node_id in executable_nodes:
                node = nodes[node_id]
                tasks.append(self._execute_node(node, context))
                pending_nodes.remove(node_id)
            
            # 노드들 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리 및 다음 노드 결정
            for node_id, result in zip(executable_nodes, results):
                executed_nodes.add(node_id)
                
                if isinstance(result, Exception):
                    logger.error(
                        "Node execution failed",
                        node_id=node_id,
                        error=str(result)
                    )
                    
                    # 폴백 정책 확인
                    if not await self._handle_node_failure(node_id, result, context):
                        raise result
                else:
                    # 다음 노드들 추가
                    next_nodes = self._get_next_nodes(
                        node_id,
                        nodes,
                        edges,
                        context
                    )
                    
                    for next_node_id in next_nodes:
                        if next_node_id not in executed_nodes:
                            pending_nodes.add(next_node_id)
    
    def _find_executable_nodes(self,
                              pending_nodes: Set[str],
                              executed_nodes: Set[str],
                              nodes: Dict[str, Any],
                              edges: Dict[str, Any]) -> List[str]:
        """실행 가능한 노드 찾기 (모든 선행 노드가 완료된 노드)"""
        
        executable = []
        
        for node_id in pending_nodes:
            if node_id not in nodes:
                continue
                
            node = nodes[node_id]
            incoming_edges = node.get('incoming', [])
            
            # 모든 선행 노드가 실행되었는지 확인
            can_execute = True
            for edge_id in incoming_edges:
                edge = edges.get(edge_id, {})
                source_node = edge.get('source')
                
                if source_node and source_node not in executed_nodes:
                    can_execute = False
                    break
            
            if can_execute:
                executable.append(node_id)
        
        return executable
    
    async def _execute_node(self,
                          node: Dict[str, Any],
                          context: ExecutionContext) -> Optional[NodeResult]:
        """단일 노드 실행"""
        
        node_id = node['id']
        node_type = node['type']
        node_config = node.get('config', {})
        
        logger.info(
            "Executing node",
            node_id=node_id,
            node_type=node_type
        )
        
        # 노드 실행 시작
        context.start_node_execution(node_id)
        
        try:
            # SLA 체크
            max_latency = context.get_policy('sla.max_latency_ms')
            
            # 노드 실행기 가져오기
            executor = get_node_executor(node_type)
            if not executor:
                raise ValueError(f"No executor found for node type: {node_type}")
            
            # 노드 실행
            if max_latency:
                # 노드별 타임아웃 적용
                result = await asyncio.wait_for(
                    executor.execute(node, context, node_config),
                    timeout=max_latency / 1000.0
                )
            else:
                result = await executor.execute(node, context, node_config)
            
            # 실행 완료
            context.complete_node_execution(
                node_id,
                result.outputs,
                result.metrics
            )
            
            return result
            
        except asyncio.TimeoutError:
            error = f"Node execution timeout"
            context.fail_node_execution(node_id, error)
            raise
            
        except Exception as e:
            error = f"Node execution failed: {str(e)}"
            context.fail_node_execution(node_id, error)
            raise
    
    def _get_next_nodes(self,
                       node_id: str,
                       nodes: Dict[str, Any],
                       edges: Dict[str, Any],
                       context: ExecutionContext) -> List[str]:
        """다음 실행할 노드 결정"""
        
        next_nodes = []
        node = nodes.get(node_id, {})
        outgoing_edges = node.get('outgoing', [])
        
        for edge_id in outgoing_edges:
            edge = edges.get(edge_id, {})
            target_node = edge.get('target')
            condition = edge.get('condition')
            
            # 조건 평가
            if condition:
                if not self._evaluate_condition(condition, context):
                    continue
            
            if target_node:
                next_nodes.append(target_node)
        
        return next_nodes
    
    def _evaluate_condition(self,
                          condition: str,
                          context: ExecutionContext) -> bool:
        """엣지 조건 평가"""
        
        try:
            # 변수 치환
            variables = context.get_all_variables()
            
            # 안전한 평가를 위한 제한된 환경
            safe_dict = {
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'True': True,
                    'False': False,
                    'None': None
                }
            }
            safe_dict.update(variables)
            
            # 조건 평가
            result = eval(condition, safe_dict)
            return bool(result)
            
        except Exception as e:
            logger.warning(
                "Condition evaluation failed",
                condition=condition,
                error=str(e)
            )
            return False
    
    async def _handle_node_failure(self,
                                  node_id: str,
                                  error: Exception,
                                  context: ExecutionContext) -> bool:
        """노드 실패 처리 (폴백 정책)"""
        
        # 폴백 정책 확인
        fallback = context.get_policy('fallback')
        if not fallback:
            return False
        
        # 에러 타입별 폴백
        if isinstance(error, TimeoutError):
            on_timeout = fallback.get('on_timeout')
            if on_timeout:
                action = on_timeout.get('action')
                if action == 'retry':
                    # 재시도 로직
                    logger.info(f"Retrying node {node_id}")
                    return True
                elif action == 'skip':
                    # 노드 스킵
                    context.skip_node_execution(node_id, "Skipped due to timeout")
                    return True
        
        # LLM 에러 폴백
        if 'llm' in node_id.lower():
            on_llm_error = fallback.get('on_llm_error')
            if on_llm_error:
                action = on_llm_error.get('action')
                if action == 'use_model':
                    # 대체 모델 사용
                    fallback_model = on_llm_error.get('model')
                    logger.info(f"Using fallback model {fallback_model}")
                    # 폴백 모델로 재실행 로직
                    return True
        
        return False
    
    def _extract_outputs(self, context: ExecutionContext) -> Dict[str, Any]:
        """실행 결과에서 출력 추출"""
        
        outputs = {}
        
        # 각 노드의 출력 수집
        for node_id, result in context.node_results.items():
            if result.status == NodeStatus.SUCCESS:
                outputs[node_id] = result.outputs
        
        # 최종 출력 변수 추가
        final_outputs = context.get_variable('final_outputs', {})
        if final_outputs:
            outputs['final'] = final_outputs
        
        # 전체 변수 중 output으로 표시된 것들
        for key, value in context.variables.items():
            if key.startswith('output.'):
                output_key = key[7:]  # 'output.' 제거
                outputs[output_key] = value
        
        return outputs
    
    async def cancel_execution(self, execution_id: str, reason: str = ""):
        """실행 중인 워크플로우 취소"""
        
        context = self.running_executions.get(execution_id)
        if context:
            context.cancel_execution(reason)
            logger.info(
                "Workflow execution cancelled",
                execution_id=execution_id,
                reason=reason
            )
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionStatus]:
        """실행 상태 조회"""
        
        context = self.running_executions.get(execution_id)
        if context:
            return context.status
        return None
    
    def get_running_executions(self) -> List[str]:
        """실행 중인 워크플로우 목록"""
        
        return list(self.running_executions.keys())