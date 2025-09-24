# src/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Kainexa Core"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API
    API_PREFIX: str = "/api/v1"
    # ALLOWED_ORIGINS를 문자열로 받아서 리스트로 변환
    ALLOWED_ORIGINS: str = "http://localhost:3000", "http://localhost:8000, *"
    
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
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """ALLOWED_ORIGINS_STR을 리스트로 변환"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS_STR.split(",")]
    
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()