# src/auth/mcp_permissions.py
"""
MCP (Model Context Protocol) 권한 모델
"""
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

class Role(Enum):
    """사용자 역할"""
    USER = "user"
    AGENT = "agent"
    DEVELOPER = "developer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class Resource(Enum):
    """리소스 타입"""
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    MODEL = "model"
    ACTION = "action"
    ANALYTICS = "analytics"
    SYSTEM = "system"

class Permission(Enum):
    """권한 타입"""
    # 기본 권한
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    
    # 특수 권한
    EXECUTE = "execute"
    INVOKE = "invoke"
    RETRIEVE = "retrieve"
    
    # 관리 권한
    GRANT = "grant"
    REVOKE = "revoke"
    AUDIT = "audit"

@dataclass
class TokenPayload:
    """JWT 토큰 페이로드"""
    user_id: str
    role: Role
    permissions: List[str]
    session_id: str
    issued_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessPolicy:
    """접근 정책"""
    resource: Resource
    permissions: Set[Permission]
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def allows(self, permission: Permission) -> bool:
        """권한 허용 여부"""
        return permission in self.permissions
    
    def check_conditions(self, context: Dict[str, Any]) -> bool:
        """조건 확인"""
        for key, value in self.conditions.items():
            if key == "ip_whitelist":
                if context.get("ip") not in value:
                    return False
            elif key == "time_window":
                current_time = datetime.now()
                start_time = datetime.fromisoformat(value["start"])
                end_time = datetime.fromisoformat(value["end"])
                if not (start_time <= current_time <= end_time):
                    return False
            elif key == "rate_limit":
                # Rate limiting logic here
                pass
        return True

class RolePermissionMatrix:
    """역할별 권한 매트릭스"""
    
    # 기본 권한 매핑
    ROLE_PERMISSIONS = {
        Role.USER: {
            Resource.CONVERSATION: {Permission.READ, Permission.WRITE},
            Resource.KNOWLEDGE: {Permission.READ},
            Resource.ANALYTICS: {Permission.READ}
        },
        Role.AGENT: {
            Resource.CONVERSATION: {Permission.READ, Permission.WRITE},
            Resource.KNOWLEDGE: {Permission.READ, Permission.RETRIEVE},
            Resource.ACTION: {Permission.INVOKE, Permission.EXECUTE},
            Resource.ANALYTICS: {Permission.READ, Permission.WRITE}
        },
        Role.DEVELOPER: {
            Resource.CONVERSATION: {Permission.READ, Permission.WRITE, Permission.DELETE},
            Resource.KNOWLEDGE: {Permission.READ, Permission.WRITE, Permission.RETRIEVE},
            Resource.MODEL: {Permission.READ, Permission.EXECUTE},
            Resource.ACTION: {Permission.INVOKE, Permission.EXECUTE},
            Resource.ANALYTICS: {Permission.READ, Permission.WRITE}
        },
        Role.ADMIN: {
            Resource.CONVERSATION: {Permission.READ, Permission.WRITE, Permission.DELETE},
            Resource.KNOWLEDGE: {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.RETRIEVE},
            Resource.MODEL: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            Resource.ACTION: {Permission.INVOKE, Permission.EXECUTE, Permission.GRANT},
            Resource.ANALYTICS: {Permission.READ, Permission.WRITE, Permission.AUDIT},
            Resource.SYSTEM: {Permission.READ, Permission.WRITE}
        },
        Role.SUPER_ADMIN: {
            # Super admin has all permissions
            resource: set(Permission) for resource in Resource
        }
    }
    
    @classmethod
    def get_permissions(cls, role: Role) -> Dict[Resource, Set[Permission]]:
        """역할에 따른 권한 조회"""
        return cls.ROLE_PERMISSIONS.get(role, {})
    
    @classmethod
    def has_permission(cls, 
                       role: Role,
                       resource: Resource,
                       permission: Permission) -> bool:
        """권한 확인"""
        role_perms = cls.get_permissions(role)
        resource_perms = role_perms.get(resource, set())
        return permission in resource_perms

class MCPAuthManager:
    """MCP 인증 관리자"""
    
    def __init__(self, 
                 secret_key: str,
                 algorithm: str = "HS256",
                 token_expire_minutes: int = 30):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire_minutes = token_expire_minutes
        self.security = HTTPBearer()
        
        # 세션 저장소 (실제로는 Redis 사용)
        self.sessions: Dict[str, TokenPayload] = {}
        
        # 권한 캐시
        self.permission_cache: Dict[str, AccessPolicy] = {}
    
    def create_token(self, 
                    user_id: str,
                    role: Role,
                    permissions: Optional[List[str]] = None) -> str:
        """JWT 토큰 생성"""
        
        # 세션 ID 생성
        session_id = secrets.token_urlsafe(32)
        
        # 토큰 페이로드
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.token_expire_minutes)
        
        # 역할 기반 권한 조회
        if permissions is None:
            role_perms = RolePermissionMatrix.get_permissions(role)
            permissions = []
            for resource, perms in role_perms.items():
                for perm in perms:
                    permissions.append(f"{resource.value}:{perm.value}")
        
        payload = TokenPayload(
            user_id=user_id,
            role=role,
            permissions=permissions,
            session_id=session_id,
            issued_at=now,
            expires_at=expires_at
        )
        
        # JWT 생성
        token_data = {
            "sub": user_id,
            "role": role.value,
            "permissions": permissions,
            "session_id": session_id,
            "iat": now.timestamp(),
            "exp": expires_at.timestamp()
        }
        
        token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        
        # 세션 저장
        self.sessions[session_id] = payload
        
        logger.info(f"Token created for user {user_id} with role {role.value}")
        
        return token
    
    def verify_token(self, token: str) -> TokenPayload:
        """토큰 검증"""
        try:
            # JWT 디코드
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # 세션 확인
            session_id = payload.get("session_id")
            if session_id not in self.sessions:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid session"
                )
            
            session = self.sessions[session_id]
            
            # 만료 확인
            if datetime.utcnow() > session.expires_at:
                del self.sessions[session_id]
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired"
                )
            
            return session
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def check_permission(self,
                        token_payload: TokenPayload,
                        resource: Resource,
                        permission: Permission,
                        context: Optional[Dict[str, Any]] = None) -> bool:
        """권한 확인"""
        
        # Super admin은 모든 권한
        if token_payload.role == Role.SUPER_ADMIN:
            return True
        
        # 역할 기반 권한 확인
        if RolePermissionMatrix.has_permission(
            token_payload.role,
            resource,
            permission
        ):
            # 추가 조건 확인
            if context:
                policy_key = f"{resource.value}:{permission.value}"
                if policy_key in self.permission_cache:
                    policy = self.permission_cache[policy_key]
                    return policy.check_conditions(context)
            return True
        
        # 명시적 권한 확인
        permission_str = f"{resource.value}:{permission.value}"
        return permission_str in token_payload.permissions
    
    def require_permission(self,
                          resource: Resource,
                          permission: Permission):
        """권한 데코레이터"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # 토큰 추출 (FastAPI 의존성에서)
                token = kwargs.get('token')
                if not token:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # 토큰 검증
                payload = self.verify_token(token)
                
                # 권한 확인
                if not self.check_permission(payload, resource, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission denied: {resource.value}:{permission.value}"
                    )
                
                # 원본 함수 실행
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def grant_permission(self,
                        granter: TokenPayload,
                        user_id: str,
                        resource: Resource,
                        permission: Permission) -> bool:
        """권한 부여"""
        
        # Grant 권한 확인
        if not self.check_permission(granter, resource, Permission.GRANT):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No grant permission"
            )
        
        # 권한 부여 로직
        logger.info(f"Permission granted: {user_id} -> {resource.value}:{permission.value}")
        return True
    
    def revoke_permission(self,
                         revoker: TokenPayload,
                         user_id: str,
                         resource: Resource,
                         permission: Permission) -> bool:
        """권한 회수"""
        
        # Revoke 권한 확인
        if not self.check_permission(revoker, resource, Permission.REVOKE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No revoke permission"
            )
        
        # 권한 회수 로직
        logger.info(f"Permission revoked: {user_id} -> {resource.value}:{permission.value}")
        return True
    
    def audit_access(self,
                    token_payload: TokenPayload,
                    resource: Resource,
                    action: str,
                    result: bool):
        """접근 감사 로그"""
        audit_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": token_payload.user_id,
            "role": token_payload.role.value,
            "session_id": token_payload.session_id,
            "resource": resource.value,
            "action": action,
            "result": "allowed" if result else "denied"
        }
        
        logger.info(f"Audit log: {audit_log}")
        # 실제로는 데이터베이스에 저장

# FastAPI 의존성
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
    auth_manager: MCPAuthManager = Depends()
) -> TokenPayload:
    """현재 사용자 조회"""
    token = credentials.credentials
    return auth_manager.verify_token(token)

async def require_role(role: Role):
    """역할 요구 의존성"""
    async def role_checker(
        current_user: TokenPayload = Depends(get_current_user)
    ):
        if current_user.role != role and current_user.role != Role.SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {role.value} required"
            )
        return current_user
    return role_checker

async def require_permission(resource: Resource, permission: Permission):
    """권한 요구 의존성"""
    async def permission_checker(
        current_user: TokenPayload = Depends(get_current_user),
        auth_manager: MCPAuthManager = Depends()
    ):
        if not auth_manager.check_permission(current_user, resource, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission {resource.value}:{permission.value} required"
            )
        return current_user
    return permission_checker