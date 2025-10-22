.PHONY: help install dev test lint format clean docker-build docker-up docker-down deploy

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
DOCKER_COMPOSE := docker-compose

# Default target
help:
	@echo "Realtor AI Copilot - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install          - Install dependencies"
	@echo "  make dev              - Run development server"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage"
	@echo "  make lint             - Run linters"
	@echo "  make format           - Format code"
	@echo "  make clean            - Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-up        - Start all services with Docker Compose"
	@echo "  make docker-down      - Stop all services"
	@echo "  make docker-logs      - View Docker logs"
	@echo "  make docker-shell     - Open shell in app container"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate       - Run database migrations"
	@echo "  make db-seed          - Seed database with test data"
	@echo "  make db-reset         - Reset database"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-staging   - Deploy to staging"
	@echo "  make deploy-prod      - Deploy to production"
	@echo "  make rollback         - Rollback deployment"
	@echo ""

# Development
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ -v --cov=app --cov-report=html --cov-report=term

test-watch:
	$(PYTEST) tests/ -v --watch

lint:
	$(FLAKE8) app/ tests/ --max-line-length=120 --ignore=E203,W503
	$(BLACK) --check app/ tests/
	$(ISORT) --check-only app/ tests/

format:
	$(BLACK) app/ tests/
	$(ISORT) app/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

# Docker
docker-build:
	docker build -t realtor-ai-copilot:latest .

docker-up:
	$(DOCKER_COMPOSE) up -d

docker-down:
	$(DOCKER_COMPOSE) down

docker-restart:
	$(DOCKER_COMPOSE) restart app

docker-logs:
	$(DOCKER_COMPOSE) logs -f app

docker-shell:
	$(DOCKER_COMPOSE) exec app /bin/bash

docker-ps:
	$(DOCKER_COMPOSE) ps

docker-clean:
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

# Database
db-migrate:
	$(PYTHON) scripts/migrate_db.py

db-seed:
	$(PYTHON) scripts/seed_data.py

db-reset:
	$(DOCKER_COMPOSE) exec db psql -U postgres -c "DROP DATABASE IF EXISTS realtor_ai;"
	$(DOCKER_COMPOSE) exec db psql -U postgres -c "CREATE DATABASE realtor_ai;"
	make db-migrate
	make db-seed

# Deployment
deploy-staging:
	./scripts/deploy.sh deploy staging

deploy-prod:
	./scripts/deploy.sh deploy production

rollback:
	./scripts/deploy.sh rollback production

# CI/CD
ci-lint:
	$(FLAKE8) app/ tests/ --max-line-length=120 --ignore=E203,W503 --exit-zero
	$(BLACK) --check app/ tests/
	$(ISORT) --check-only app/ tests/

ci-test:
	$(PYTEST) tests/ -v --cov=app --cov-report=xml --cov-report=term

ci-build:
	docker build -t realtor-ai-copilot:test .

# Monitoring
logs-app:
	tail -f logs/app.log

logs-docker:
	$(DOCKER_COMPOSE) logs -f

metrics:
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

# Security
security-scan:
	safety check
	bandit -r app/ -f json

# Documentation
docs-serve:
	@echo "API Documentation: http://localhost:8000/docs"
	@echo "ReDoc: http://localhost:8000/redoc"
