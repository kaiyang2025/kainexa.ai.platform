# src/api/routes/workflow.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
import yaml
import json

from src.core.database import get_db
from src.orchestration.dsl_parser import DSLParser

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

@router.get("/")
async def list_workflows(db: AsyncSession = Depends(get_db)):
    """워크플로우 목록 조회"""
    query = "SELECT * FROM workflows ORDER BY created_at DESC"
    result = await db.execute(query)
    workflows = result.fetchall()
    
    return [
        {
            "id": w.id,
            "name": w.name,
            "version": w.version,
            "status": w.status,
            "created_at": w.created_at,
            "updated_at": w.updated_at
        }
        for w in workflows
    ]

@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """워크플로우 상세 조회"""
    query = "SELECT * FROM workflows WHERE id = :id"
    result = await db.execute(query, {"id": workflow_id})
    workflow = result.fetchone()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "content": workflow.content,  # YAML content
        "version": workflow.version,
        "status": workflow.status
    }

@router.post("/")
async def create_workflow(
    workflow_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """워크플로우 생성"""
    
    # YAML 유효성 검증
    parser = DSLParser()
    try:
        graph_config = parser.parse_yaml(workflow_data['content'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid workflow: {e}")
    
    # DB 저장
    query = """
        INSERT INTO workflows (name, content, version, status)
        VALUES (:name, :content, :version, :status)
        RETURNING id
    """
    
    result = await db.execute(query, {
        "name": workflow_data['name'],
        "content": workflow_data['content'],
        "version": workflow_data.get('version', '1.0'),
        "status": "draft"
    })
    
    workflow_id = result.scalar()
    await db.commit()
    
    return {"id": workflow_id, "message": "Workflow created successfully"}

@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: str,
    workflow_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """워크플로우 수정"""
    
    # YAML 유효성 검증
    parser = DSLParser()
    try:
        graph_config = parser.parse_yaml(workflow_data['content'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid workflow: {e}")
    
    # DB 업데이트
    query = """
        UPDATE workflows 
        SET name = :name, content = :content, version = :version, updated_at = NOW()
        WHERE id = :id
    """
    
    await db.execute(query, {
        "id": workflow_id,
        "name": workflow_data['name'],
        "content": workflow_data['content'],
        "version": workflow_data.get('version', '1.0')
    })
    
    await db.commit()
    
    return {"message": "Workflow updated successfully"}

@router.post("/{workflow_id}/deploy")
async def deploy_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """워크플로우 배포"""
    
    # 워크플로우 조회
    query = "SELECT * FROM workflows WHERE id = :id"
    result = await db.execute(query, {"id": workflow_id})
    workflow = result.fetchone()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # 배포 프로세스
    # 1. 이전 버전 백업
    # 2. 새 버전 활성화
    # 3. 캐시 갱신
    
    update_query = """
        UPDATE workflows 
        SET status = 'active', deployed_at = NOW()
        WHERE id = :id
    """
    
    await db.execute(update_query, {"id": workflow_id})
    await db.commit()
    
    return {"message": "Workflow deployed successfully"}

@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db)
):
    """워크플로우 삭제"""
    
    # Soft delete
    query = """
        UPDATE workflows 
        SET status = 'deleted', deleted_at = NOW()
        WHERE id = :id
    """
    
    await db.execute(query, {"id": workflow_id})
    await db.commit()
    
    return {"message": "Workflow deleted successfully"}

@router.post("/{workflow_id}/test")
async def test_workflow(
    workflow_id: str,
    test_input: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
):
    """워크플로우 테스트 실행"""
    
    # 워크플로우 로드
    query = "SELECT content FROM workflows WHERE id = :id"
    result = await db.execute(query, {"id": workflow_id})
    workflow = result.fetchone()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # 테스트 실행
    from src.orchestration.dsl_parser import DSLParser
    from src.orchestration.graph_executor import GraphExecutor, ExecutionContext
    
    parser = DSLParser()
    graph_config = parser.parse_yaml(workflow.content)
    
    context = ExecutionContext(
        session_id=f"test_{workflow_id}",
        variables=test_input
    )
    
    executor = GraphExecutor()
    result_context = await executor.execute_graph(graph_config, context)
    
    return {
        "execution_history": result_context.history,
        "final_variables": result_context.variables,
        "errors": result_context.errors
    }