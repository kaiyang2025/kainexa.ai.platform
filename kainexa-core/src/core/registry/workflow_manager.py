# src/core/registry/workflow_manager.py
"""
Workflow Registry Manager
Handles workflow DSL upload, versioning, and management
"""
import hashlib
import json
import yaml
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from uuid import UUID, uuid4
import asyncio
import asyncpg
from pydantic import BaseModel, Field, validator
import structlog
from enum import Enum

logger = structlog.get_logger()

# ========== Enums ==========
class WorkflowStatus(str, Enum):
    UPLOADED = "uploaded"
    COMPILING = "compiling" 
    COMPILED = "compiled"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class Environment(str, Enum):
    DEV = "dev"
    STAGE = "stage"
    PROD = "prod"

class NodeType(str, Enum):
    INTENT = "intent"
    LLM = "llm"
    API = "api"
    CONDITION = "condition"
    LOOP = "loop"

# ========== Models ==========
class WorkflowMetadata(BaseModel):
    """Workflow metadata model"""
    author: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class WorkflowDSL(BaseModel):
    """Workflow DSL structure"""
    version: str
    workflow: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    policies: Optional[Dict[str, Any]] = Field(default_factory=dict)
    environments: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('version')
    def validate_version(cls, v):
        """Validate semantic versioning"""
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError(f"Invalid version format: {v}. Use semantic versioning (e.g., 1.0.0)")
        return v
    
    @validator('nodes')
    def validate_nodes(cls, v):
        """Validate node structure"""
        for node in v:
            if 'id' not in node or 'type' not in node:
                raise ValueError("Each node must have 'id' and 'type'")
            if node['type'] not in [t.value for t in NodeType]:
                raise ValueError(f"Invalid node type: {node['type']}")
        return v

class WorkflowVersion(BaseModel):
    """Workflow version model"""
    id: UUID
    workflow_id: UUID
    version: str
    dsl_raw: str
    dsl_format: str
    compiled_graph: Optional[Dict[str, Any]] = None
    status: WorkflowStatus
    checksums: Dict[str, str]
    created_by: str
    created_at: datetime
    compiled_at: Optional[datetime] = None

class Workflow(BaseModel):
    """Workflow model"""
    id: UUID
    namespace: str
    name: str
    description: Optional[str] = None
    created_by: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    active_versions: Dict[str, str] = Field(default_factory=dict)  # env -> version

# ========== Registry Manager ==========
class WorkflowManager:
    """Workflow Registry Manager"""
    
    def __init__(self, db_pool=None):
        """
        db_pool이 없으면 테스트/로컬 실행을 위해 in-memory 모드로 동작합니다.
        실제 운영에선 db_pool을 주입해 사용하세요.
        """
        self.db_pool = db_pool
        # in-memory 보조 구조(필요 시 내부 메서드에서 사용; 기존 코드에 영향 없음)
        self._mem_store = {
            "workflows": {},   # (workflow_id, version) -> dsl/json 등
            "compiled": {},    # (workflow_id, version) -> compiled artifact
            "published": set(),# (workflow_id, version)
            "metrics": {},     # workflow_id -> {runs, failures, ...}
        }
        
    # ========== DSL Operations ==========
    async def upload_dsl(self, 
                        dsl_content: str,
                        format: str,
                        user_id: str) -> WorkflowVersion:
        """Upload and store workflow DSL"""
        
        logger.info("Uploading workflow DSL", user=user_id, format=format)
        
        # 1. Parse DSL
        try:
            if format == 'yaml':
                dsl_dict = yaml.safe_load(dsl_content)
            else:
                dsl_dict = json.loads(dsl_content)
                
            parsed = WorkflowDSL(**dsl_dict)
        except Exception as e:
            logger.error(f"Failed to parse DSL: {e}")
            raise ValueError(f"Invalid DSL format: {e}")
        
        # 2. Calculate checksums
        checksums = {
            'dsl': hashlib.sha256(dsl_content.encode()).hexdigest(),
            'nodes': hashlib.sha256(json.dumps(parsed.nodes, sort_keys=True).encode()).hexdigest(),
            'edges': hashlib.sha256(json.dumps(parsed.edges, sort_keys=True).encode()).hexdigest()
        }
        
        # 3. Get or create workflow
        workflow = await self.get_or_create_workflow(
            namespace=parsed.workflow['namespace'],
            name=parsed.workflow['name'],
            description=parsed.workflow.get('metadata', {}).get('description'),
            user_id=user_id
        )
        
        # 4. Check for duplicate version
        existing = await self.get_version(workflow.id, parsed.workflow['version'])
        if existing:
            raise ValueError(f"Version {parsed.workflow['version']} already exists")
        
        # 5. Create version record
        version = await self.create_version(
            workflow_id=workflow.id,
            version=parsed.workflow['version'],
            dsl_raw=dsl_content,
            dsl_format=format,
            checksums=checksums,
            created_by=user_id
        )
        
        logger.info(
            "Workflow DSL uploaded successfully",
            workflow_id=str(workflow.id),
            version=version.version
        )
        
        return version
    
    async def get_or_create_workflow(self,
                                    namespace: str,
                                    name: str,
                                    description: Optional[str],
                                    user_id: str) -> Workflow:
        """Get existing workflow or create new one"""
        
        async with self.db.acquire() as conn:
            # Try to get existing
            row = await conn.fetchrow("""
                SELECT * FROM workflows 
                WHERE namespace = $1 AND name = $2 AND deleted_at IS NULL
            """, namespace, name)
            
            if row:
                return Workflow(**dict(row))
            
            # Create new workflow
            row = await conn.fetchrow("""
                INSERT INTO workflows (namespace, name, description, created_by)
                VALUES ($1, $2, $3, $4)
                RETURNING *
            """, namespace, name, description, user_id)
            
            return Workflow(**dict(row))
    
    async def create_version(self,
                            workflow_id: UUID,
                            version: str,
                            dsl_raw: str,
                            dsl_format: str,
                            checksums: Dict[str, str],
                            created_by: str) -> WorkflowVersion:
        """Create new workflow version"""
        
        async with self.db.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO workflow_versions 
                (workflow_id, version, dsl_raw, dsl_format, checksums, created_by, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
            """, workflow_id, version, dsl_raw, dsl_format, 
                json.dumps(checksums), created_by, WorkflowStatus.UPLOADED.value)
            
            return WorkflowVersion(**dict(row))
    
    async def get_version(self, 
                         workflow_id: UUID, 
                         version: str) -> Optional[WorkflowVersion]:
        """Get specific workflow version"""
        
        async with self.db.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM workflow_versions
                WHERE workflow_id = $1 AND version = $2
            """, workflow_id, version)
            
            return WorkflowVersion(**dict(row)) if row else None
    
    async def list_versions(self, workflow_id: UUID) -> List[WorkflowVersion]:
        """List all versions of a workflow"""
        
        async with self.db.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM workflow_versions
                WHERE workflow_id = $1
                ORDER BY created_at DESC
            """, workflow_id)
            
            return [WorkflowVersion(**dict(row)) for row in rows]
    
    # ========== Compilation ==========
    async def compile_workflow(self, 
                              workflow_id: UUID,
                              version: str) -> Dict[str, Any]:
        """Compile workflow DSL to execution graph"""
        
        logger.info("Compiling workflow", workflow_id=str(workflow_id), version=version)
        
        # 1. Get version
        version_obj = await self.get_version(workflow_id, version)
        if not version_obj:
            raise ValueError(f"Version {version} not found")
        
        # 2. Update status to compiling
        await self.update_version_status(workflow_id, version, WorkflowStatus.COMPILING)
        
        try:
            # 3. Parse DSL
            if version_obj.dsl_format == 'yaml':
                dsl = yaml.safe_load(version_obj.dsl_raw)
            else:
                dsl = json.loads(version_obj.dsl_raw)
            
            # 4. Validate and compile
            compiled_graph = await self._compile_dsl_to_graph(dsl)
            
            # 5. Store compiled graph
            async with self.db.acquire() as conn:
                await conn.execute("""
                    UPDATE workflow_versions
                    SET compiled_graph = $1, 
                        status = $2,
                        compiled_at = $3
                    WHERE workflow_id = $4 AND version = $5
                """, json.dumps(compiled_graph), WorkflowStatus.COMPILED.value,
                    datetime.utcnow(), workflow_id, version)
            
            logger.info("Compilation successful", workflow_id=str(workflow_id), version=version)
            return compiled_graph
            
        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            await self.update_version_status(workflow_id, version, WorkflowStatus.FAILED)
            raise
    
    async def _compile_dsl_to_graph(self, dsl: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DSL to executable graph"""
        
        nodes_map = {}
        edges_map = {}
        
        # 1. Process nodes
        for node in dsl['nodes']:
            node_id = node['id']
            nodes_map[node_id] = {
                'id': node_id,
                'type': node['type'],
                'config': node.get('config', {}),
                'position': node.get('position', {'x': 0, 'y': 0}),
                'incoming': [],
                'outgoing': []
            }
        
        # 2. Process edges
        for edge in dsl['edges']:
            edge_id = edge.get('id', f"{edge['source']}-{edge['target']}")
            edges_map[edge_id] = {
                'id': edge_id,
                'source': edge['source'],
                'target': edge['target'],
                'condition': edge.get('condition'),
                'label': edge.get('label')
            }
            
            # Update node connections
            if edge['source'] in nodes_map:
                nodes_map[edge['source']]['outgoing'].append(edge_id)
            if edge['target'] in nodes_map:
                nodes_map[edge['target']]['incoming'].append(edge_id)
        
        # 3. Find entry points (nodes with no incoming edges)
        entry_points = [
            node_id for node_id, node in nodes_map.items()
            if len(node['incoming']) == 0
        ]
        
        # 4. Validate graph
        if not entry_points:
            raise ValueError("No entry point found in graph")
        
        # 5. Build compiled graph
        compiled_graph = {
            'version': dsl['version'],
            'metadata': dsl['workflow'],
            'nodes': nodes_map,
            'edges': edges_map,
            'entry_points': entry_points,
            'policies': dsl.get('policies', {}),
            'environments': dsl.get('environments', {}),
            'compiled_at': datetime.utcnow().isoformat()
        }
        
        return compiled_graph
    
    async def update_version_status(self,
                                   workflow_id: UUID,
                                   version: str,
                                   status: WorkflowStatus):
        """Update workflow version status"""
        
        async with self.db.acquire() as conn:
            await conn.execute("""
                UPDATE workflow_versions
                SET status = $1
                WHERE workflow_id = $2 AND version = $3
            """, status.value, workflow_id, version)
    
    # ========== Environment Management ==========
    async def publish_workflow(self,
                              workflow_id: UUID,
                              version: str,
                              environment: Environment,
                              user_id: str) -> Dict[str, Any]:
        """Publish workflow to environment"""
        
        logger.info(
            "Publishing workflow",
            workflow_id=str(workflow_id),
            version=version,
            environment=environment.value
        )
        
        # 1. Verify version is compiled
        version_obj = await self.get_version(workflow_id, version)
        if not version_obj:
            raise ValueError(f"Version {version} not found")
        
        if version_obj.status != WorkflowStatus.COMPILED:
            raise ValueError(f"Version {version} is not compiled")
        
        # 2. Begin transaction for atomic update
        async with self.db.acquire() as conn:
            async with conn.transaction():
                # Get current active version
                current = await conn.fetchrow("""
                    SELECT * FROM env_routes
                    WHERE workflow_id = $1 AND environment = $2
                """, workflow_id, environment.value)
                
                if current:
                    # Update existing route
                    await conn.execute("""
                        UPDATE env_routes
                        SET active_version = $1,
                            version_id = $2,
                            activated_at = $3,
                            activated_by = $4,
                            previous_version = $5
                        WHERE workflow_id = $6 AND environment = $7
                    """, version, version_obj.id, datetime.utcnow(), user_id,
                        current['active_version'], workflow_id, environment.value)
                else:
                    # Create new route
                    await conn.execute("""
                        INSERT INTO env_routes 
                        (workflow_id, environment, active_version, version_id, 
                         activated_at, activated_by)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, workflow_id, environment.value, version, version_obj.id,
                        datetime.utcnow(), user_id)
                
                # Log deployment
                await conn.execute("""
                    INSERT INTO deployment_history
                    (workflow_id, version_id, version, environment, 
                     action, deployed_by)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, workflow_id, version_obj.id, version, environment.value,
                    'deploy', user_id)
        
        # 3. Get workflow info for endpoint
        workflow = await self.get_workflow(workflow_id)
        
        result = {
            'workflow_id': str(workflow_id),
            'namespace': workflow.namespace,
            'name': workflow.name,
            'version': version,
            'environment': environment.value,
            'endpoint': f"/workflow/{workflow.namespace}/{workflow.name}/execute",
            'activated_at': datetime.utcnow().isoformat(),
            'activated_by': user_id
        }
        
        logger.info("Workflow published successfully", **result)
        return result
    
    async def rollback_workflow(self,
                               workflow_id: UUID,
                               environment: Environment,
                               user_id: str) -> Dict[str, Any]:
        """Rollback to previous version"""
        
        async with self.db.acquire() as conn:
            # Get current route
            route = await conn.fetchrow("""
                SELECT * FROM env_routes
                WHERE workflow_id = $1 AND environment = $2
            """, workflow_id, environment.value)
            
            if not route or not route['previous_version']:
                raise ValueError("No previous version to rollback to")
            
            previous_version = route['previous_version']
            
            # Get version object
            version_obj = await self.get_version(workflow_id, previous_version)
            if not version_obj:
                raise ValueError(f"Previous version {previous_version} not found")
            
            # Update route
            async with conn.transaction():
                await conn.execute("""
                    UPDATE env_routes
                    SET active_version = $1,
                        version_id = $2,
                        activated_at = $3,
                        activated_by = $4,
                        previous_version = $5,
                        rollback_count = rollback_count + 1
                    WHERE workflow_id = $6 AND environment = $7
                """, previous_version, version_obj.id, datetime.utcnow(),
                    user_id, route['active_version'], workflow_id, environment.value)
                
                # Log rollback
                await conn.execute("""
                    INSERT INTO deployment_history
                    (workflow_id, version_id, version, environment,
                     action, deployed_by, deployment_metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, workflow_id, version_obj.id, previous_version, environment.value,
                    'rollback', user_id, json.dumps({
                        'from_version': route['active_version'],
                        'to_version': previous_version
                    }))
        
        return {
            'workflow_id': str(workflow_id),
            'environment': environment.value,
            'rolled_back_to': previous_version,
            'rolled_back_from': route['active_version'],
            'rollback_count': route['rollback_count'] + 1,
            'performed_by': user_id,
            'performed_at': datetime.utcnow().isoformat()
        }
    
    # ========== Query Operations ==========
    async def get_workflow(self, workflow_id: UUID) -> Workflow:
        """Get workflow by ID"""
        
        async with self.db.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT w.*,
                    jsonb_object_agg(
                        er.environment,
                        er.active_version
                    ) FILTER (WHERE er.active_version IS NOT NULL) as active_versions
                FROM workflows w
                LEFT JOIN env_routes er ON w.id = er.workflow_id
                WHERE w.id = $1 AND w.deleted_at IS NULL
                GROUP BY w.id
            """, workflow_id)
            
            if not row:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            return Workflow(**dict(row))
    
    async def get_active_workflow(self,
                                 namespace: str,
                                 name: str,
                                 environment: Environment) -> Optional[Dict[str, Any]]:
        """Get active workflow for execution"""
        
        async with self.db.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    w.id,
                    w.namespace,
                    w.name,
                    wv.version,
                    wv.compiled_graph,
                    er.environment
                FROM workflows w
                JOIN env_routes er ON w.id = er.workflow_id
                JOIN workflow_versions wv ON 
                    er.workflow_id = wv.workflow_id 
                    AND er.active_version = wv.version
                WHERE w.namespace = $1 
                    AND w.name = $2 
                    AND er.environment = $3
                    AND w.deleted_at IS NULL
            """, namespace, name, environment.value)
            
            if not row:
                return None
            
            return {
                'id': row['id'],
                'namespace': row['namespace'],
                'name': row['name'],
                'version': row['version'],
                'environment': row['environment'],
                'compiled_graph': row['compiled_graph']
            }
    
    async def list_workflows(self,
                            namespace: Optional[str] = None,
                            page: int = 1,
                            limit: int = 20) -> Dict[str, Any]:
        """List workflows with pagination"""
        
        offset = (page - 1) * limit
        
        async with self.db.acquire() as conn:
            # Build query
            query = """
                SELECT 
                    w.*,
                    jsonb_object_agg(
                        er.environment,
                        jsonb_build_object(
                            'version', er.active_version,
                            'activated_at', er.activated_at
                        )
                    ) FILTER (WHERE er.active_version IS NOT NULL) as environments
                FROM workflows w
                LEFT JOIN env_routes er ON w.id = er.workflow_id
                WHERE w.deleted_at IS NULL
            """
            
            params = []
            if namespace:
                query += " AND w.namespace = $1"
                params.append(namespace)
            
            query += " GROUP BY w.id ORDER BY w.created_at DESC"
            
            # Add pagination
            if namespace:
                query += f" LIMIT ${len(params)+1} OFFSET ${len(params)+2}"
            else:
                query += " LIMIT $1 OFFSET $2"
            params.extend([limit, offset])
            
            # Execute query
            rows = await conn.fetch(query, *params)
            
            # Get total count
            count_query = "SELECT COUNT(*) FROM workflows WHERE deleted_at IS NULL"
            if namespace:
                count_query += " AND namespace = $1"
                total = await conn.fetchval(count_query, namespace)
            else:
                total = await conn.fetchval(count_query)
        
        workflows = [Workflow(**dict(row)) for row in rows]
        
        return {
            'workflows': [w.dict() for w in workflows],
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'total_pages': (total + limit - 1) // limit
            }
        }
    
    # ========== Cleanup Operations ==========
    async def delete_workflow(self, workflow_id: UUID, user_id: str):
        """Soft delete workflow"""
        
        async with self.db.acquire() as conn:
            await conn.execute("""
                UPDATE workflows
                SET deleted_at = $1
                WHERE id = $2
            """, datetime.utcnow(), workflow_id)
        
        logger.info("Workflow deleted", workflow_id=str(workflow_id), deleted_by=user_id)