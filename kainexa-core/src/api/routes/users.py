# src/api/routes/users.py
from fastapi import APIRouter, Header, HTTPException

router = APIRouter(prefix="/api/v1", tags=["users"])

@router.get("/me")
async def me(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != "test-jwt-token":
        raise HTTPException(status_code=401, detail="invalid token")
    return {"username": "test_user", "roles": ["user"]}
