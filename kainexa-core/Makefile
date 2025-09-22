# Makefile
.PHONY: help setup install dev test clean docker-up docker-down migrate

# 가상 환경 경로
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn
PYTEST = $(VENV)/bin/pytest
ALEMBIC = $(VENV)/bin/alembic

help:
	@echo "Available commands:"
	@echo "  setup      Create virtual environment and install dependencies"
	@echo "  install    Install dependencies (requires venv)"
	@echo "  dev        Start development environment"
	@echo "  test       Run tests"
	@echo "  docker-up  Start Docker services"
	@echo "  docker-down Stop Docker services"
	@echo "  clean      Clean up containers and cache"
	@echo "  migrate    Run database migrations"

setup:
	@chmod +x scripts/setup_env.sh
	@./scripts/setup_env.sh

install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ -f requirements-dev.txt ]; then \
		$(PIP) install -r requirements-dev.txt; \
	fi

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

docker-up:
	docker-compose up -d postgres redis qdrant
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services are up!"

docker-down:
	docker-compose down

migrate: $(VENV)/bin/activate
	$(ALEMBIC) upgrade head

dev: $(VENV)/bin/activate docker-up
	$(UVICORN) src.api.main:app --reload --host 0.0.0.0 --port 8000

test: $(VENV)/bin/activate
	$(PYTEST) tests/ -v --cov=src

clean:
	docker-compose down -v
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov