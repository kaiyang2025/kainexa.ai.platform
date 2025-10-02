# -*- coding: utf-8 -*-
"""
Lightweight auth shims for tests.

- verify_token(token: str) -> dict
- get_current_user(...) -> dict (FastAPI Depends용)

실서비스 JWT 검증은 추후 교체.
"""

from __future__ import annotations
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# FastAPI Security dependency (bearer token)
_security = HTTPBearer(auto_error=False)


def verify_token(token: Optional[str]) -> Dict:
    """
    테스트/개발용 토큰 검증 대행.
    - 토큰이 없으면 401
    - 있으면 더미 사용자 정보 반환
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # TODO: 실제 JWT 검증 로직으로 교체
    # ex) jwt.decode(token, SECRET, algorithms=[ALGO])
    return {
        "sub": "test@kainexa.local",
        "email": "test@kainexa.local",
        "roles": ["tester"],
        "token": token,
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> Dict:
    """
    FastAPI dependency. Authorization: Bearer <token> 헤더에서 토큰 추출.
    """
    token = credentials.credentials if credentials else None
    return verify_token(token)


__all__ = ["verify_token", "get_current_user"]
