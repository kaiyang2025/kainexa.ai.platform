# src/api/routes/workflow_routes.py
"""
Kainexa Core API Routes for Workflow Management
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
import yaml
import json
import asyncpg
from pydantic import BaseModel, Field

from src.core.registry.workflow_manager import (
    WorkflowManager, 
    WorkflowStatus, 
    Environment,
    Workflow,
    WorkflowVersion
)
from src.core.auth import verify_token, get_current_user
from src.core.database import get_db_pool

router = APIRouter(prefix="/api/v1", tags=["workflows"])

# ========== Request/Response Models ==========
class WorkflowUploadRequest(BaseModel):
    """DSL upload request"""
    content: str
    format: str = Field(default="yaml", pattern="^(yaml|json)$")

class CompileRequest(BaseModel):
    """Compile request"""
    workflow_id: UUID
    version: str

class SimulateRequest(BaseModel):
    """Simulation request"""
    workflow_id: UUID
    version: str
    input: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

class PublishRequest(BaseModel):
    """Publish request"""
    workflow_id: UUID
    version: str
    environment: Environment

class ExecutionRequest(BaseModel):
    """Execution request"""
    session_id: str
    input: Dict[str, Any]
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

# ========== Dependency Injection ==========
async def get_workflow_manager(db_pool: asyncpg.Pool = Depends(get_db_pool)) -> WorkflowManager:
    """Get workflow manager instance"""
    return WorkflowManager(db_pool)

# ========== Workflow Management Endpoints ==========
@router.post("/workflows", status_code=201)
async def upload_workflow(
    file: Optional[UploadFile] = File(None),
    request: Optional[WorkflowUploadRequest] = None,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """
    Upload workflow DSL in YAML or JSON format
    
    Can accept either:
    - File upload (multipart/form-data)
    - Raw content in request body (application/json)
    """
    try:
        # Determine content source
        if file:
            content = await file.read()
            content = content.decode('utf-8')
            format = 'yaml' if file.filename.endswith('.yaml') or file.filename.endswith('.yml') else 'json'
        elif request:
            content = request.content
            format = request.format
        else:
            raise HTTPException(400, "No workflow content provided")
        
        # Upload to registry
        version = await manager.upload_dsl(
            dsl_content=content,
            format=format,
            user_id=current_user['email']
        )
        
        return {
            'workflow_id': str(version.workflow_id),
            'version': version.version,
            'status': version.status,
            'message': 'Workflow uploaded successfully',
            'checksums': version.checksums
        }
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to upload workflow: {str(e)}")

@router.get("/workflows")
async def list_workflows(
    namespace: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """List workflows with pagination"""
    try:
        result = await manager.list_workflows(
            namespace=namespace,
            page=page,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Failed to list workflows: {str(e)}")

@router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: UUID,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Get workflow details"""
    try:
        workflow = await manager.get_workflow(workflow_id)
        return workflow.dict()
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to get workflow: {str(e)}")

@router.get("/workflows/{workflow_id}/versions")
async def list_workflow_versions(
    workflow_id: UUID,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """List all versions of a workflow"""
    try:
        versions = await manager.list_versions(workflow_id)
        return {
            'workflow_id': str(workflow_id),
            'versions': [
                {
                    'version': v.version,
                    'status': v.status,
                    'created_at': v.created_at.isoformat(),
                    'created_by': v.created_by,
                    'compiled_at': v.compiled_at.isoformat() if v.compiled_at else None
                }
                for v in versions
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list versions: {str(e)}")

@router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: UUID,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Delete workflow (soft delete)"""
    try:
        await manager.delete_workflow(workflow_id, current_user['email'])
        return {'message': 'Workflow deleted successfully'}
    except Exception as e:
        raise HTTPException(500, f"Failed to delete workflow: {str(e)}")

# ========== Compilation Endpoints ==========
@router.post("/workflows/compile")
async def compile_workflow(
    request: CompileRequest,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Compile workflow DSL to execution graph"""
    try:
        compiled_graph = await manager.compile_workflow(
            workflow_id=request.workflow_id,
            version=request.version
        )
        
        return {
            'workflow_id': str(request.workflow_id),
            'version': request.version,
            'status': 'success',
            'compiled_graph': compiled_graph,
            'compiled_at': datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                'workflow_id': str(request.workflow_id),
                'version': request.version,
                'status': 'failed',
                'errors': [str(e)]
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Compilation failed: {str(e)}")

# ========== Simulation Endpoints ==========
@router.post("/workflows/simulate")
async def simulate_workflow(
    request: SimulateRequest,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Simulate workflow execution with sample input"""
    try:
        # Get compiled workflow
        version = await manager.get_version(request.workflow_id, request.version)
        if not version or version.status != WorkflowStatus.COMPILED:
            raise ValueError("Workflow must be compiled before simulation")
        
        # Import simulator (to be implemented)
        from src.core.simulator import WorkflowSimulator
        
        simulator = WorkflowSimulator()
        result = await simulator.simulate(
            compiled_graph=version.compiled_graph,
            input_data=request.input,
            context=request.context or {}
        )
        
        return {
            'workflow_id': str(request.workflow_id),
            'version': request.version,
            'execution_path': result['path'],
            'outputs': result['outputs'],
            'metrics': result['metrics']
        }
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {str(e)}")

# ========== Publishing Endpoints ==========
@router.post("/workflows/publish")
async def publish_workflow(
    request: PublishRequest,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Publish workflow to environment"""
    try:
        # Check permission
        if request.environment == Environment.PROD and current_user['role'] not in ['admin', 'publisher']:
            raise HTTPException(403, "Insufficient permissions to publish to production")
        
        result = await manager.publish_workflow(
            workflow_id=request.workflow_id,
            version=request.version,
            environment=request.environment,
            user_id=current_user['email']
        )
        
        return {
            'workflow_id': str(result['workflow_id']),
            'version': result['version'],
            'environment': result['environment'],
            'endpoint': result['endpoint'],
            'status': 'active',
            'activated_at': result['activated_at'],
            'activated_by': result['activated_by']
        }
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Publishing failed: {str(e)}")

@router.post("/workflows/activate")
async def activate_workflow(
    request: PublishRequest,
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Activate specific workflow version (for rollback)"""
    try:
        result = await manager.publish_workflow(
            workflow_id=request.workflow_id,
            version=request.version,
            environment=request.environment,
            user_id=current_user['email']
        )
        
        return {
            'workflow_id': str(result['workflow_id']),
            'version': result['version'],
            'environment': result['environment'],
            'status': 'activated',
            'message': f"Version {result['version']} activated in {result['environment']}"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Activation failed: {str(e)}")

@router.post("/workflows/{workflow_id}/rollback")
async def rollback_workflow(
    workflow_id: UUID,
    environment: Environment = Body(...),
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Rollback to previous version"""
    try:
        result = await manager.rollback_workflow(
            workflow_id=workflow_id,
            environment=environment,
            user_id=current_user['email']
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Rollback failed: {str(e)}")

# ========== Execution Endpoints ==========
@router.post("/workflow/{namespace}/{name}/execute")
async def execute_workflow(
    namespace: str,
    name: str,
    request: ExecutionRequest,
    environment: Environment = Query(Environment.PROD),
    manager: WorkflowManager = Depends(get_workflow_manager),
    current_user: Dict = Depends(get_current_user)
):
    """Execute workflow"""
    try:
        # Get active workflow
        workflow = await manager.get_active_workflow(
            namespace=namespace,
            name=name,
            environment=environment
        )
        
        if not workflow:
            raise HTTPException(404, f"No active workflow found for {namespace}/{name} in {environment}")
        
        # Import executor (to be implemented in next phase)
        from src.core.executor import GraphExecutor, ExecutionContext
        
        executor = GraphExecutor()
        
        # Build execution context
        context = ExecutionContext(
            session_id=request.session_id,
            user_id=current_user['email'],
            tenant_id=current_user.get('tenant_id'),
            channel=request.context.get('channel', 'api'),
            language=request.context.get('language', 'ko'),
            variables=request.input
        )
        
        # Execute workflow
        result = await executor.execute(
            graph=workflow['compiled_graph'],
            input_data=request.input,
            context=context
        )
        
        return {
            'execution_id': result.execution_id,
            'status': result.status,
            'outputs': result.outputs,
            'metrics': result.metrics,
            'trace_url': f"/api/v1/executions/{result.execution_id}/trace"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Execution failed: {str(e)}")

@router.get("/executions/{execution_id}")
async def get_execution(
    execution_id: UUID,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    current_user: Dict = Depends(get_current_user)
):
    """Get execution details"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM executions
                WHERE execution_id = $1
            """, execution_id)
            
            if not row:
                raise HTTPException(404, "Execution not found")
            
            return dict(row)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get execution: {str(e)}")

@router.get("/executions/{execution_id}/trace")
async def get_execution_trace(
    execution_id: UUID,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    current_user: Dict = Depends(get_current_user)
):
    """Get execution trace"""
    try:
        async with db_pool.acquire() as conn:
            # Get execution
            execution = await conn.fetchrow("""
                SELECT * FROM executions
                WHERE execution_id = $1
            """, execution_id)
            
            if not execution:
                raise HTTPException(404, "Execution not found")
            
            # Get node executions
            nodes = await conn.fetch("""
                SELECT * FROM node_executions
                WHERE execution_id = $1
                ORDER BY started_at
            """, execution_id)
            
            return {
                'execution_id': str(execution_id),
                'workflow_id': str(execution['workflow_id']),
                'version': execution['version'],
                'status': execution['status'],
                'started_at': execution['started_at'].isoformat(),
                'completed_at': execution['completed_at'].isoformat() if execution['completed_at'] else None,
                'total_duration_ms': execution['latency_ms'],
                'spans': [
                    {
                        'span_id': str(node['id']),
                        'node_id': node['node_id'],
                        'node_type': node['node_type'],
                        'status': node['status'],
                        'started_at': node['started_at'].isoformat() if node['started_at'] else None,
                        'duration_ms': node['duration_ms'],
                        'inputs': node['inputs'],
                        'outputs': node['outputs'],
                        'error': node['error']
                    }
                    for node in nodes
                ]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get execution trace: {str(e)}")

# ========== Health Check ==========
@router.get("/health")
async def health_check():
    """API health check"""
    return {
        'status': 'healthy',
        'service': 'workflow-api',
        'timestamp': datetime.utcnow().isoformat()
    }