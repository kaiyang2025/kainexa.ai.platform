# Makefile
.PHONY: help setup install dev test clean docker-up docker-down migrate monitor gpu-setup gpu-run

# 가상 환경 경로
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn
PYTEST = $(VENV)/bin/pytest
ALEMBIC = $(VENV)/bin/alembic

help:
	@echo "Available commands:"
	@echo ""
	@echo "  === Basic Setup ==="
	@echo "  setup           Create virtual environment and install dependencies"
	@echo "  install         Install dependencies (requires venv)"
	@echo "  migrate         Run database migrations"
	@echo ""
	@echo "  === Development ==="
	@echo "  dev             Start development environment"
	@echo "  test            Run tests"
	@echo "  clean           Clean up containers and cache"
	@echo ""
	@echo "  === Docker Services ==="
	@echo "  docker-up       Start all Docker services"
	@echo "  docker-down     Stop all Docker services"
	@echo "  monitor-up      Start monitoring stack (Prometheus, Grafana, etc.)"
	@echo "  monitor-down    Stop monitoring stack"
	@echo ""
	@echo "  === GPU Operations ==="
	@echo "  gpu-setup       Setup GPU environment"
	@echo "  gpu-run         Run tensor parallel inference"
	@echo "  gpu-benchmark   Run GPU benchmarks"
	@echo ""
	@echo "  === Combined Operations ==="
	@echo "  full-stack      Start complete v2 stack with monitoring"
	@echo "  status          Check all services status"

setup:
	@chmod +x scripts/setup_env.sh
	@./scripts/setup_env.sh
	@echo "Installing additional v2 requirements..."
	$(PIP) install deepspeed==0.12.0 pynvml accelerate prometheus-client clickhouse-driver opentelemetry-api

install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ -f requirements-dev.txt ]; then \
		$(PIP) install -r requirements-dev.txt; \
	fi
	@if [ -f requirements-gpu.txt ]; then \
		$(PIP) install -r requirements-gpu.txt; \
	fi

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

# Docker 서비스 관리
docker-up:
	docker compose up -d postgres redis qdrant
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Core services are up!"

docker-down:
	docker compose down

# 모니터링 스택
monitor-up:
	docker compose -f docker-compose.monitoring.yml up -d
	@echo "Monitoring stack starting..."
	@sleep 10
	@echo "Monitoring available at:"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/kainexa123)"
	@echo "  - Jaeger: http://localhost:16686"

monitor-down:
	docker-compose -f docker-compose.monitoring.yml down

# GPU 작업
gpu-setup:
	@chmod +x scripts/run_gpu_parallel.sh
	@echo "Checking GPU environment..."
	@nvidia-smi
	@echo "Setting up NVLink..."
	@nvidia-smi topo -m
	@echo "GPU setup complete!"

gpu-run:
	@./scripts/run_gpu_parallel.sh parallel

gpu-benchmark:
	@./scripts/run_gpu_parallel.sh benchmark

# 데이터베이스
migrate: $(VENV)/bin/activate
	$(ALEMBIC) upgrade head
	@echo "Running ClickHouse migrations..."
	docker exec -i kainexa-clickhouse clickhouse-client < monitoring/clickhouse/init.sql

# 개발 서버
dev: $(VENV)/bin/activate docker-up
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

# 전체 v2 스택 실행
full-stack: docker-up monitor-up migrate
	@echo "Starting full v2 stack..."
	@docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
	@sleep 10
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 &
	@echo ""
	@echo "=== Kainexa v2 Stack Running ==="
	@echo "API: http://localhost:8000/api/v1/docs"
	@echo "Grafana: http://localhost:3000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"
	@echo "================================"

# 상태 확인
status:
	@echo "=== Service Status ==="
	@echo -n "PostgreSQL: "
	@docker-compose exec -T postgres pg_isready -U kainexa &>/dev/null && echo "✅ Running" || echo "❌ Not running"
	@echo -n "Redis: "
	@docker-compose exec -T redis redis-cli ping &>/dev/null && echo "✅ Running" || echo "❌ Not running"
	@echo -n "Qdrant: "
	@curl -s http://localhost:6333/health &>/dev/null && echo "✅ Running" || echo "❌ Not running"
	@echo -n "ClickHouse: "
	@curl -s http://localhost:8123/ping &>/dev/null && echo "✅ Running" || echo "❌ Not running"
	@echo -n "Prometheus: "
	@curl -s http://localhost:9090/-/healthy &>/dev/null && echo "✅ Running" || echo "❌ Not running"
	@echo -n "Grafana: "
	@curl -s http://localhost:3000/api/health &>/dev/null && echo "✅ Running" || echo "❌ Not running"
	@echo -n "API: "
	@curl -s http://localhost:8000/api/v1/health &>/dev/null && echo "✅ Running" || echo "❌ Not running"
	@echo ""
	@echo "=== GPU Status ==="
	@nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || echo "No GPU detected"

# 테스트
test: $(VENV)/bin/activate
	$(PYTEST) tests/ -v --cov=src

# 정리
clean:
	docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down -v
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf logs/*.log