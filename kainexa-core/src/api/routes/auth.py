# src/api/routes/auth.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/login", response_model=LoginResponse)
async def login(_: LoginRequest):
    # 테스트용 더미 토큰 반환
    return LoginResponse(access_token="test-jwt-token")
