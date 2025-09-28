# src/core/config.py  (pydantic v2 기준)
from __future__ import annotations

import json
from typing import Optional, List

from pydantic_settings import BaseSettings, SettingsConfigDict 
from pydantic import computed_field

class Settings(BaseSettings):
    
    # .env 읽기 설정 + 모르는 키는 무시(추가 입력 금지 에러 방지)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    
    # Application
    APP_NAME: str = "Kainexa Core"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
            
    # API
    API_PREFIX: str = "/api/v1"
    
    # .env에 있는 ALLOWED_ORIGINS_STR을 정식 필드로 정의
    # 예) "http://localhost:3000,http://localhost:8000,*"
    ALLOWED_ORIGINS_STR: str = "http://localhost:3000,http://localhost:8000,http://192.168.1.215:8000,*"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://kainexa:password@localhost:5432/kainexa_db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL: int = 3600
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    ALGORITHM: str = "HS256"

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    # Logging
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[str] = None
       
    # pydantic v2 권장: 계산 필드로 ALLOWED_ORIGINS 제공(필드/프로퍼티 충돌 제거)
    @computed_field(return_type=List[str])
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """
        - 콤마 구분 문자열: "http://a.com, http://b.com, *"
        - JSON 배열도 허용: '["http://a.com","http://b.com","*"]'
        - 비어있으면 ["*"] 반환
        """
        s = (self.ALLOWED_ORIGINS_STR or "").strip()
        if not s:
            return ["*"]
        if s.startswith("["):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
        return [p.strip() for p in s.split(",") if p.strip()]

settings = Settings()