# Makefile

.PHONY: help install dev test clean

help:
	@echo "Available commands:"
	@echo "  install    Install dependencies"
	@echo "  dev        Start development environment"
	@echo "  test       Run tests"
	@echo "  clean      Clean up containers and cache"

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

dev:
	./scripts/dev.sh

test:
	pytest tests/ -v --cov=src

clean:
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete