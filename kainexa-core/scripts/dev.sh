# scripts/dev.sh
#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Kainexa Core development environment...${NC}"

# 가상 환경 확인
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# 가상 환경 활성화
source venv/bin/activate

# .env 파일 확인
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}.env file not found. Copying from .env.example...${NC}"
        cp .env.example .env
        echo -e "${YELLOW}Please update .env with your configuration.${NC}"
    else
        echo -e "${RED}.env file not found and no .env.example available.${NC}"
        exit 1
    fi
fi

# Docker 확인
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Docker 서비스 시작
echo -e "${GREEN}Starting Docker services...${NC}"
docker-compose up -d postgres redis qdrant

# 서비스 헬스 체크
echo -e "${GREEN}Checking service health...${NC}"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker-compose exec -T postgres pg_isready -U kainexa > /dev/null 2>&1; then
        echo -e "${GREEN}PostgreSQL is ready!${NC}"
        break
    fi
    echo -e "${YELLOW}Waiting for PostgreSQL... (attempt $((attempt+1))/$max_attempts)${NC}"
    sleep 2
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}PostgreSQL failed to start.${NC}"
    exit 1
fi

# Redis 체크
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}Redis is ready!${NC}"
else
    echo -e "${RED}Redis is not responding.${NC}"
    exit 1
fi

# Alembic 설치 확인 및 마이그레이션
if command -v alembic &> /dev/null; then
    echo -e "${GREEN}Running database migrations...${NC}"
    alembic upgrade head
else
    echo -e "${YELLOW}Alembic not found. Skipping migrations.${NC}"
fi

# API 서버 시작
echo -e "${GREEN}Starting API server...${NC}"
echo -e "${GREEN}API Documentation: http://localhost:8000/api/v1/docs${NC}"
echo -e "${GREEN}Health Check: http://localhost:8000/api/v1/health${NC}"

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000