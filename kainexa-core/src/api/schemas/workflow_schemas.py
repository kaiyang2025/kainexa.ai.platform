from __future__ import annotations

from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class WorkflowUploadRequest(BaseModel):
    """워크플로우 DSL 업로드 요청"""
    namespace: str
    name: str
    dsl_content: str  # JSON 문자열(테스트에서 json.dumps(sample_dsl)로 전달)


class WorkflowCompileRequest(BaseModel):
    """업로드된 워크플로우를 컴파일 요청"""
    workflow_id: str
    version: str


class WorkflowSimulateRequest(BaseModel):
    """컴파일된 워크플로우를 시뮬레이션 실행 요청"""
    workflow_id: str
    version: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    user_context: Dict[str, Any] = Field(default_factory=dict)


class WorkflowPublishRequest(BaseModel):
    """컴파일된 워크플로우를 특정 환경에 퍼블리시 요청"""
    workflow_id: str
    version: str
    environment: Literal["development", "staging", "production"] = "development"


__all__ = [
    "WorkflowUploadRequest",
    "WorkflowCompileRequest",
    "WorkflowSimulateRequest",
    "WorkflowPublishRequest",
]
