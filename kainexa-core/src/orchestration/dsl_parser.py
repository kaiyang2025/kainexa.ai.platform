# src/orchestration/dsl_parser.py
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger()

class StepType(Enum):
    """실행 단계 타입"""
    INTENT_CLASSIFY = "intent_classify"
    RETRIEVE_KNOWLEDGE = "retrieve_knowledge"
    LLM_GENERATE = "llm_generate"
    RESPONSE_POSTPROCESS = "response_postprocess"
    EXTERNAL_API = "external_api"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    TRANSFORM = "transform"
    CACHE_CHECK = "cache_check"

class PolicyType(Enum):
    """정책 타입"""
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    FALLBACK = "fallback"
    ESCALATION = "escalation"
    RETRY = "retry"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    COST_LIMIT = "cost_limit"

@dataclass
class StepConfig:
    """실행 단계 설정"""
    name: str
    type: StepType
    params: Dict[str, Any] = field(default_factory=dict)
    policy: Optional[Dict[str, Any]] = None
    next_steps: List[str] = field(default_factory=list)
    parallel: bool = False
    timeout: float = 30.0
    retry: int = 0
    cache: bool = False
    
@dataclass
class GraphConfig:
    """실행 그래프 설정"""
    name: str
    version: str = "1.0"
    steps: Dict[str, StepConfig] = field(default_factory=dict)
    entry_point: str = "start"
    policies: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DSLParser:
    """DSL YAML/JSON 파서"""
    
    def __init__(self):
        self.validators = {
            StepType.INTENT_CLASSIFY: self._validate_intent_step,
            StepType.RETRIEVE_KNOWLEDGE: self._validate_retrieve_step,
            StepType.LLM_GENERATE: self._validate_llm_step,
            StepType.EXTERNAL_API: self._validate_api_step,
            StepType.CONDITION: self._validate_condition_step,
            StepType.PARALLEL: self._validate_parallel_step
        }
        
    def parse_yaml(self, yaml_content: str) -> GraphConfig:
        """YAML DSL 파싱"""
        try:
            config_dict = yaml.safe_load(yaml_content)
            return self.parse_dict(config_dict)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML: {e}")
            raise ValueError(f"Invalid YAML format: {e}")
            
    def parse_json(self, json_content: str) -> GraphConfig:
        """JSON DSL 파싱"""
        try:
            config_dict = json.loads(json_content)
            return self.parse_dict(config_dict)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
            
    def parse_dict(self, config_dict: Dict[str, Any]) -> GraphConfig:
        """딕셔너리를 GraphConfig로 변환"""
        
        # 기본 정보 추출
        graph_config = GraphConfig(
            name=config_dict.get('name', 'unnamed_graph'),
            version=config_dict.get('version', '1.0'),
            entry_point=config_dict.get('entry_point', 'start'),
            policies=config_dict.get('policies', {}),
            metadata=config_dict.get('metadata', {})
        )
        
        # 그래프 단계 파싱
        graph_dict = config_dict.get('graph', [])
        if isinstance(graph_dict, list):
            # 리스트 형식의 순차 실행
            graph_config.steps = self._parse_sequential_steps(graph_dict)
        elif isinstance(graph_dict, dict):
            # 딕셔너리 형식의 명시적 그래프
            graph_config.steps = self._parse_explicit_graph(graph_dict)
        else:
            raise ValueError("Graph must be a list or dict")
            
        # 유효성 검증
        self._validate_graph(graph_config)
        
        return graph_config
        
    def _parse_sequential_steps(self, steps_list: List[Dict]) -> Dict[str, StepConfig]:
        """순차 실행 단계 파싱"""
        steps = {}
        
        for i, step_dict in enumerate(steps_list):
            step_name = step_dict.get('step', f'step_{i}')
            step_type = self._parse_step_type(step_dict.get('type', 'unknown'))
            
            # 정책 파싱
            policy = self._parse_policy(step_dict.get('policy'))
            
            # 다음 단계 설정 (순차 실행)
            next_steps = []
            if i < len(steps_list) - 1:
                next_step_name = steps_list[i + 1].get('step', f'step_{i+1}')
                next_steps = [next_step_name]
            
            step_config = StepConfig(
                name=step_name,
                type=step_type,
                params=step_dict.get('params', {}),
                policy=policy,
                next_steps=next_steps,
                parallel=step_dict.get('parallel', False),
                timeout=step_dict.get('timeout', 30.0),
                retry=step_dict.get('retry', 0),
                cache=step_dict.get('cache', False)
            )
            
            steps[step_name] = step_config
            
        return steps
        
    def _parse_explicit_graph(self, graph_dict: Dict) -> Dict[str, StepConfig]:
        """명시적 그래프 구조 파싱"""
        steps = {}
        
        for step_name, step_dict in graph_dict.items():
            step_type = self._parse_step_type(step_dict.get('type'))
            
            step_config = StepConfig(
                name=step_name,
                type=step_type,
                params=step_dict.get('params', {}),
                policy=self._parse_policy(step_dict.get('policy')),
                next_steps=step_dict.get('next', []),
                parallel=step_dict.get('parallel', False),
                timeout=step_dict.get('timeout', 30.0),
                retry=step_dict.get('retry', 0),
                cache=step_dict.get('cache', False)
            )
            
            steps[step_name] = step_config
            
        return steps
        
    def _parse_step_type(self, type_str: str) -> StepType:
        """문자열을 StepType으로 변환"""
        try:
            return StepType(type_str.lower())
        except ValueError:
            logger.warning(f"Unknown step type: {type_str}, using LLM_GENERATE")
            return StepType.LLM_GENERATE
            
    def _parse_policy(self, policy_dict: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """정책 파싱"""
        if not policy_dict:
            return None
            
        parsed_policy = {}
        
        # 조건문 파싱
        if 'if' in policy_dict:
            condition = policy_dict['if']
            then_action = policy_dict.get('then', {})
            else_action = policy_dict.get('else', {})
            
            parsed_policy['condition'] = {
                'if': self._parse_condition(condition),
                'then': then_action,
                'else': else_action
            }
            
        # 기타 정책
        for key in ['fallback', 'escalate', 'retry', 'timeout', 'rate_limit']:
            if key in policy_dict:
                parsed_policy[key] = policy_dict[key]
                
        return parsed_policy if parsed_policy else None
        
    def _parse_condition(self, condition_str: str) -> Dict[str, Any]:
        """조건문 파싱"""
        # 간단한 조건문 파싱 (예: "confidence < 0.7")
        parts = condition_str.split()
        if len(parts) == 3:
            return {
                'variable': parts[0],
                'operator': parts[1],
                'value': self._parse_value(parts[2])
            }
        return {'raw': condition_str}
        
    def _parse_value(self, value_str: str) -> Union[str, int, float, bool]:
        """값 파싱"""
        # 숫자 변환
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
            
        # 불린 변환
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
            
        # 문자열 반환
        return value_str
        
    def _validate_graph(self, graph_config: GraphConfig):
        """그래프 유효성 검증"""
        
        # 진입점 확인
        if graph_config.entry_point not in graph_config.steps:
            if graph_config.steps:
                # 첫 번째 단계를 진입점으로 설정
                graph_config.entry_point = list(graph_config.steps.keys())[0]
            else:
                raise ValueError("Graph has no steps")
                
        # 각 단계 유효성 검증
        for step_name, step in graph_config.steps.items():
            # 단계별 검증
            if step.type in self.validators:
                self.validators[step.type](step)
                
            # 다음 단계 참조 확인
            for next_step in step.next_steps:
                if next_step not in graph_config.steps and next_step != 'end':
                    logger.warning(f"Step '{step_name}' references unknown next step '{next_step}'")
                    
    def _validate_intent_step(self, step: StepConfig):
        """의도 분류 단계 검증"""
        required_params = ['model', 'threshold']
        for param in required_params:
            if param not in step.params:
                step.params[param] = self._get_default_param(param)
                
    def _validate_retrieve_step(self, step: StepConfig):
        """지식 검색 단계 검증"""
        if 'k' not in step.params:
            step.params['k'] = 5
        if 'threshold' not in step.params:
            step.params['threshold'] = 0.7
            
    def _validate_llm_step(self, step: StepConfig):
        """LLM 생성 단계 검증"""
        if 'model' not in step.params:
            step.params['model'] = 'default'
        if 'temperature' not in step.params:
            step.params['temperature'] = 0.7
            
    def _validate_api_step(self, step: StepConfig):
        """외부 API 단계 검증"""
        required_params = ['url', 'method']
        for param in required_params:
            if param not in step.params:
                raise ValueError(f"External API step '{step.name}' missing required param '{param}'")
                
    def _validate_condition_step(self, step: StepConfig):
        """조건 단계 검증"""
        if 'condition' not in step.params:
            raise ValueError(f"Condition step '{step.name}' missing condition")
        if 'true_branch' not in step.params or 'false_branch' not in step.params:
            raise ValueError(f"Condition step '{step.name}' missing branches")
            
    def _validate_parallel_step(self, step: StepConfig):
        """병렬 실행 단계 검증"""
        if 'branches' not in step.params:
            raise ValueError(f"Parallel step '{step.name}' missing branches")
            
    def _get_default_param(self, param_name: str) -> Any:
        """기본 파라미터 값 반환"""
        defaults = {
            'model': 'default',
            'threshold': 0.5,
            'temperature': 0.7,
            'k': 5,
            'timeout': 30.0,
            'retry': 0
        }
        return defaults.get(param_name)